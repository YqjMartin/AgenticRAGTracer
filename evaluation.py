import json
import re
import string
import time
import requests
from qwen_agent.llm import get_chat_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import numpy as np
import os
MAX_WORKERS = 50
MODEL_ID = ''
API_BASE =  ''
API_URL = ''
API_KEY = ''
DEFAULT_MODEL = 'gpt-4o-mini'
INPUT_JSONL_LIST = []

def _openai_chat_api(messages, model=DEFAULT_MODEL, timeout=120):
    payload = {
        "model": model, 
        "messages": messages
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")

def _clean_json_block(item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

def run_llm_prompt(prompt, developer_prompt=None, model=DEFAULT_MODEL, return_json=False, max_retries=3):
    messages = []
    if developer_prompt:
        messages.append({"role": "system", "content": developer_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for _ in range(max_retries):
        try:
            resp_text = _openai_chat_api(messages, model=model)
            if return_json:
                resp_text = _clean_json_block(resp_text)
                try:
                    return json.loads(resp_text)
                except:
                    try:
                        if "[" in resp_text and "]" in resp_text:
                            json_part = resp_text[resp_text.index("["):resp_text.rindex("]") + 1]
                            return json.loads(json_part)
                        elif "{" in resp_text and "}" in resp_text:
                            json_part = resp_text[resp_text.index("{"):resp_text.rindex("}") + 1]
                            obj_data = json.loads(json_part)
                            return [obj_data]
                    except:
                        raise ValueError(f"Failed to parse JSON:\n{resp_text}")
            return resp_text
        except Exception as e:
            last_error = e
            time.sleep(5)
    raise RuntimeError(f"run_llm_prompt failed: {last_error}")

LLM_JUDGE_PROMPT = '''You are an expert evaluator. Evaluate whether the OTHER ANSWER preserves **all essential information** in the GOLDEN ANSWER, **with respect to the QUESTION**.
# Scoring Criteria
- **2 points** → OTHER ANSWER is fully equivalent to the GOLDEN ANSWER. Same meaning, even if reworded or paraphrased. No missing or incorrect information.
- **1 point** → OTHER ANSWER includes ALL key information from the GOLDEN ANSWER but adds **extra non-contradictory information** that may not be strictly necessary but is still valid in context.
- **0 points** → OTHER ANSWER is **missing** critical information from the GOLDEN ANSWER or introduces **incorrect/contradictory** information, based on the QUESTION.

Always consider what the QUESTION is asking when judging whether information is essential.

# ✅ Positive Examples

## ✅ 2 points

- Question: What year did the war end?  
    Golden: 1848  
    Other: The year was 1848.

- Question: Who became the first African American U.S. president?  
    Golden: Barack Obama  
    Other: Obama

- Question: When did the battle begin?  
    Golden: The battle began in 1775.  
    Other: The conflict started in the year 1775.

- Question: Which field of Nobel Prize did Marie Curie receive?
    Golden: the Nobel Prize in Physics.
    Other: Physics.

## ✅ 1 point

- Question: What is the cause of death of Mercedesz Henger's father?  
    Golden: diabetes mellitus.  
    Other: Diabetes mellitus type 2.

- Question: Who became the first African American U.S. president?  
    Golden: Barack Obama  
    Other: Barack Obama, the 44th president of the United States.

- Question: Where is the Eiffel Tower located?  
    Golden: The Eiffel Tower is in Paris.  
    Other: The Eiffel Tower is in Paris, the capital of France.

## ❌ Negative Examples (0 points)

- Question: How much is the price?  
    Golden: 50  
    Other: 25 dollars

- Question: When did the war end?  
    Golden: 1848  
    Other: In 1846

- Question: Where is the Eiffel Tower located?  
    Golden: The Eiffel Tower is in Paris.  
    Other: The Eiffel Tower is in Berlin.

- Question: Who became the first African American U.S. president?  
    Golden: Barack Obama  
    Other: Barack Obama and Abraham Lincoln were presidents during the same era.

# Output Format:
Return ONLY JSON, no extra text.
```json
{
    "answer_reason": "reason for the score",
    "answer_score": 0/1/2
}
```'''

def llm_judge(question, golden_answer, other_answer, llm_judge_prompt, num_parallel_predictions=1, model=DEFAULT_MODEL):
    prompt = f"""Input:\nQuestion: {question}\nGolden answer: {golden_answer}\nOther answer: {other_answer}"""

    def _judge_once():
        result = run_llm_prompt(prompt, developer_prompt=llm_judge_prompt, model=model, return_json=True)
        if result is None:
            return None
        return {
            "answer_score": result.get("answer_score", 0),
            "answer_reason": result.get("answer_reason", "")
        }

    results = []
    with ThreadPoolExecutor(max_workers=num_parallel_predictions) as exe:
        for r in exe.map(lambda _: _judge_once(), range(num_parallel_predictions)):
            if r is not None:
                results.append(r)

    if not results:
        return {
            "avg_score": 0,
            "reasons": [],
            "raw_scores": []
        }

    avg_score = sum(r["answer_score"] for r in results) / len(results)
    reasons = [r["answer_reason"] for r in results]
    raw_scores = [r["answer_score"] for r in results]

    return {
        "avg_score": avg_score,
        "reasons": reasons,
        "raw_scores": raw_scores
    }

def normalize_answer(s: str) -> str:
    if s.strip() in ["A", "B", "C", "D", "E"]:
        return s.strip().upper()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truths) -> float:
    if prediction is None or ground_truths is None:
        return 0.0

    if prediction.startswith("I cannot answer this question"):
        return 0.0

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    max_f1 = 0.0

    for ground_truth in ground_truths:
        if ground_truth is None:
            continue

        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] or normalized_ground_truth in ["yes", "no", "noanswer"]:
            if normalized_prediction != normalized_ground_truth:
                continue

        pred_tokens = normalized_prediction.split()
        gold_tokens = normalized_ground_truth.split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)

    return max_f1

def em_score(prediction: str, ground_truths) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for golden_answer in ground_truths:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1.0
            break
    return score

def Final_Answer(answer: str) -> str:
    return answer

def RAG_search(query: str, topk: int = 5) -> str:
        try:
            response = requests.post(
                "http://localhost:8000/retrieve",
                json={"query": query, "topk": topk},
                timeout=1200
            )
            data = response.json()
            if not data.get('results'):
                return f"No results found. Server says: {data.get('message')}"

            formatted = [
                f"[Doc: \"{doc['contents']}\"\n"
                for doc in data["results"]
            ]
            
            return (
                f"[RAGTool Results | query: \"{query}\" | topk = {topk}]\n\n" +
                "\n\n".join(formatted)
            )
        except Exception as e:
            return (
                "[RAGTool Error] An exception occurred during retrieval.\n"
                f"Error Message: {str(e)}\n"
            )

def parse_tool_calls(text: str) -> dict:
    m = re.search(r'(?s)<tool_calls>\s*(.*?)\s*</tool_calls>', text)
    return json.loads(m.group(1)) if m else {}

def test(task):
    llm = get_chat_model({
        'model': MODEL_ID,
        'model_server': API_BASE,
        'api_key': API_KEY
    })

    # Step 1: send the conversation and available functions to the model
    messages = [{
        'role': 'user',
        'content': f'''You are a world expert at making efficient plans to solve any task using an RAG Search tool. 
Now for the given task, develop a step-by-step high-level plan.
Each step should be a concise action that uses the RAG search tool, specifying the keyword you want to search for.
Do not skip steps, do not add any superfluous steps. Only write the high-level plan, DO NOT Execute TOOL CALLS in this planning step.
Each step should be clearly numbered.

Here is your task:
{task['final_question']}

Now begin! Write your plan below.'''
    }]

    functions = [{
        'name': 'RAG_search',
        'description': 'Use this tool to search relevant documents based on the input query.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The input query string to search relevant documents for.',
                },
                'topk': {
                    'type': 'integer',
                    'description': 'The number of top relevant documents to return.',
                },
            },
            'required': ['query', 'topk'],
        },
    },{
        'name': 'Final_Answer',
        'description': 'Use this tool to provide the final concise answer to the original question, based on all the information you have gathered.',
        'parameters': {
            'type': 'object',
            'properties': {
                'answer': {
                    'type': 'string',
                    'description': 'The final concise answer to the original question. Provide A WORD OR PHRASE, NOT A WHOLE SENTENCE.',
                },
            },
            'required': ['answer'],
        }
    }
    ]
    last_error = ""
    for _ in range(3):
        try:
            #print('# Assistant Planning:')
            responses = []    
            responses = llm.chat(
                messages=messages,
                functions=functions,
                stream=False,
                extra_generate_cfg=dict(
                    function_choice='none',
                ),
            )
            print(responses)
            messages.extend(responses)
            steps = 0
            total_topk = 0
            max_no_call = 0
            call_tool = True
            while(steps < 10 and max_no_call < 5):
                #print(f'\n# Assistant Response {steps}:')
                if call_tool:
                    new_message = {
                        'role': 'user',
                        'content': '''Based on the plan and the search results before (if there is), first analyse what information you have gained and what other information you still need, then Execute ONLY ONE MORE step using the RAG search tool. 
            You should never assume or invent any search results that are not explicitly provided in the context.
            If there is no search results before, just write 'Currently, no results are available.' directly instead of guessing and directly Execute the first step in your plan using the RAG search tool.
            If you find there is nothing more to search, do not search for the same or similar thing you have searched again.
            Instead, directly provide the final concise answer to the original question using the Final_Answer tool.
            When providing the final answer, DO NOT include any other content such as search results or any thoughts, just the concise final answer itself.''',
                    }
                    messages.append(new_message)
                    responses = llm.chat(
                        messages=messages,
                        functions=functions,
                        stream=False,
                    )
                else:
                    responses = llm.chat(
                        messages=messages,
                        functions=functions,
                        stream=False,
                    )
                print(responses)
                messages.extend(responses)
                # Step 2: check if the model wanted to call a function
                fncall_msgs = [rsp for rsp in responses if rsp.get('function_call', None)]
                #print('# Function Call Messages:', fncall_msgs)
                if fncall_msgs: 
                    call_tool = True
                    max_no_call = 0
                    if fncall_msgs[0]['function_call']['name'] != 'Final_Answer':
                        # Note: the JSON response may not always be valid; be sure to handle errors
                        steps += 1
                        available_functions = {
                            'RAG_search': RAG_search,
                            'Final_Answer': Final_Answer,
                        }

                        for msg in fncall_msgs:
                            function_name = msg['function_call']['name']
                            function_to_call = available_functions[function_name]
                            function_args = json.loads(msg['function_call']['arguments'])
                            function_response = function_to_call(
                                query=function_args.get('query'),
                                topk=function_args.get('topk', 5),
                            )
                            #print(f'# function_response: {function_response}')
                            total_topk += function_args.get('topk', 5)
                            messages.append({
                                'role': 'function',
                                'name': function_name,
                                'content': function_response,
                            })
                    else:
                        pred_answer = None
                        for msg in fncall_msgs:
                            if msg['function_call']['name'] == 'Final_Answer':
                                function_args = json.loads(msg['function_call']['arguments'])
                                pred_answer = function_args.get('answer', None)
                        if pred_answer is None:
                            pred_answer = responses[-1]['content']
                        f1 = f1_score(pred_answer, task['final_answer'])
                        em = em_score(pred_answer, task['final_answer'])
                        llm_score = llm_judge(
                            question=task['final_question'],
                            golden_answer=task['final_answer'],
                            other_answer=pred_answer,
                            llm_judge_prompt=LLM_JUDGE_PROMPT,
                            num_parallel_predictions=1,
                        )
                        return {
                            'task': task,
                            'question': task['final_question'],
                            'golden_answer': task['final_answer'],
                            'pred_answer': pred_answer,
                            'f1_score': f1,
                            'em_score': em,
                            'llm_score': llm_score,
                            'agent_step_number': steps,
                            'topk_per_step': total_topk / steps if steps > 0 else 0,
                            'EssEq_results': llm_score,
                            'trajectory': messages
                        }
                else:
                    call_tool = False
                    max_no_call += 1
                    continue
                
            if steps >= 10:
                print("over 10 steps.")
                return {
                    'task': task,
                    'question': task['final_question'],
                    'golden_answer': task['final_answer'],
                    'pred_answer': "over 10 steps.",
                    'f1_score': 0,
                    'em_score': 0,
                    'llm_score': 0,
                    'agent_step_number': 0,
                    'topk_per_step': 0,
                    'EssEq_results': 0,
                    'trajectory': messages
                }
            else:
                print("over 5 response no call tool.")
                return {
                    'task': task,
                    'question': task['final_question'],
                    'golden_answer': task['final_answer'],
                    'pred_answer': "over 5 response no call tool.",
                    'f1_score': 0,
                    'em_score': 0,
                    'llm_score': 0,
                    'agent_step_number': 0,
                    'topk_per_step': 0,
                    'EssEq_results': 0,
                    'trajectory': messages
                }
        except Exception as e:
            print(f"[ERROR]: {e}")
            last_error = e
            time.sleep(5)
            continue
    print(f"[ERROR Final]: {last_error}")
    return {
        'task': task,
        'question': task['final_question'],
        'golden_answer': task['final_answer'],
        'pred_answer': f"[ERROR]: {last_error}",
        'f1_score': 0,
        'em_score': 0,
        'llm_score': 0,
        'agent_step_number': 0,
        'topk_per_step': 0,
        'EssEq_results': 0,
        'trajectory': messages
    }

def calculate_average_scores(results):
    scores = {
        'em_scores': [],
        'f1_scores': [],
        'agent_step_numbers': [],
        'agent_step_numbers_correct': [],
        'agent_step_numbers_wrong': [],
        'llm_judge_score': [],
        'topks_per_step': [],
        'topks_per_step_correct': [],
        'topks_per_step_wrong': [],
    }
    
    for result in results:
        scores['em_scores'].append(result.get('em_score'))
        scores['f1_scores'].append(result.get('f1_score'))
        scores['agent_step_numbers'].append(result.get('agent_step_number'))
        scores['topks_per_step'].append(result.get('topk_per_step'))
        llm_judge_score = 1 if result.get('EssEq_results').get("avg_score") >= 1 else 0
        if result.get('EssEq_results').get("avg_score") >= 1:
            scores['agent_step_numbers_correct'].append(result.get('agent_step_number'))
            scores['topks_per_step_correct'].append(result.get('topk_per_step'))
        else:
            scores['agent_step_numbers_wrong'].append(result.get('agent_step_number'))
            scores['topks_per_step_wrong'].append(result.get('topk_per_step'))
        scores['llm_judge_score'].append(llm_judge_score)
    
    averages = {
        'average_em_score': np.mean(scores['em_scores']),
        'average_f1_score': np.mean(scores['f1_scores']),
        'average_agent_step_number': np.mean(scores['agent_step_numbers']),
        'average_agent_step_number_correct': np.mean(scores['agent_step_numbers_correct']) if scores['agent_step_numbers_correct'] else 0,
        'average_agent_step_number_wrong': np.mean(scores['agent_step_numbers_wrong']) if scores['agent_step_numbers_wrong'] else 0,
        'average_topk_per_step': np.mean(scores['topks_per_step']),
        'average_topk_per_step_correct': np.mean(scores['topks_per_step_correct']) if scores['topks_per_step_correct'] else 0,
        'average_topk_per_step_wrong': np.mean(scores['topks_per_step_wrong']) if scores['topks_per_step_wrong'] else 0,
        'average_llm_judge_score': np.mean(scores['llm_judge_score']),
        'total_samples': len(results)
    }
    
    return averages

if __name__ == '__main__':
    output_path_list = []
    for input_path in INPUT_JSONL_LIST[:]:
        output_path = input_path.replace("final_data", f"llm_results/{MODEL_ID}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(input_path, 'r', encoding='utf-8') as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        print(f"处理文件: {input_path}")
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(test, task): task for task in tasks}
            for future in tqdm(as_completed(future_map), total=len(tasks), desc="Processing"):
                results.append(future.result())
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"结果已保存至: {output_path}")
        output_path_list.append(output_path)
        
    for input_path in output_path_list[:]:
        print(f"正在处理文件 '{input_path}'...")
        results = []
        with open(input_path, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        averages = calculate_average_scores(results)
        final_result_path = input_path.replace(".jsonl", "_metric.jsonl")
        with open(final_result_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(averages, ensure_ascii=False, indent=2))
        print(f"处理完成，结果已保存至 '{final_result_path}'")
