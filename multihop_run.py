import os
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
from multihop_pipeline import *
import random

# 初始化日志
os.makedirs("./Logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"./Logs/{timestamp}.log"

logger = logging.getLogger("multihop_gen")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("日志系统初始化完成")

# ===== 并行控制参数 =====
counter_lock = Lock()
total_input = 0
total_success = 0

multihop_pipeline = MultiHopPipeline(
    model_id="gpt-4o-mini",
    retriever_url="http://localhost:8000/retrieve",
    prompt_file="multihop_prompt.yaml"
)

def process_item(sample):

    global total_input, total_success

    id = sample.get("id")
    question = sample.get("question")
    answers = sample.get("refined_answer")
    original_doc = sample.get("contents") if sample.get("contents") else sample.get("doc")

    if not question or not answers or not original_doc:
        logger.warning(f"[跳过] {id} 缺少字段")
        return [], []

    if isinstance(answers, list):
        answer = answers[0]
    else:
        answer = answers

    with counter_lock:
        total_input += 1

    try:
        results = multihop_pipeline.process_sample_morehop(
            original_question=question,
            original_answer=answer,
            original_doc=original_doc,
            topk=10,
            gen_qa_num=5,
            pattern='full',
            num_hop=4,
            every_hop_qa_num=15,
        )
        vaild_result = results.get("valid_results", [])
        full_result = results.get("full_results", [])
        if vaild_result:
            with counter_lock:
                total_success += len(vaild_result)
            logger.info(f"[成功] {id} -> 生成 {len(vaild_result)} 条 multihop QA")
        else:
            logger.info(f"[无结果] {id}")
        return vaild_result, full_result

    except Exception as e:
        logger.error(f"[异常] {id} 错误: {e}", exc_info=True)
        return [], []


def run_multihop_parallel(input_path, output_path,
                          max_workers, limit):

    with open(input_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    if limit:
        random.seed(42) 
        all_data = random.sample(all_data, limit)
        #all_data = all_data[:limit]

    logger.info(f"加载数据 {len(all_data)} 条")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vaild_results = []
    full_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, sample) for sample in all_data]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", dynamic_ncols=True):
            vaild_result, full_result = fut.result()
            if vaild_result:
                vaild_results.extend(vaild_result)
            if full_result:
                full_results.extend(full_result)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in vaild_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    full_output_path = output_path.replace(".jsonl", "_full.jsonl")
    with open(full_output_path, "w", encoding="utf-8") as f:
        for item in full_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    data_hop4 = []
    data_hop3 = []
    data_hop2 = []
    for item in full_results:
        if item.get('hop_4', []):
            data_hop4.append(item)
        if not item.get('hop_4', []) and item.get('hop_3', []):
            data_hop3.append(item)
        if not item.get('hop_4', []) and not item.get('hop_3', []) and item.get('hop_2', []):
            data_hop2.append(item)

    outfile_hop2 = output_path.replace('.jsonl', '_hop2.jsonl')
    with open(outfile_hop2, 'w', encoding='utf-8') as f_out:
        for item in data_hop2:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Filtered data with hop2 only, total {len(data_hop2)} items.")

    outfile_hop3 = output_path.replace('.jsonl', '_hop3.jsonl')
    with open(outfile_hop3, 'w', encoding='utf-8') as f_out:
        for item in data_hop3:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Filtered data with hop3 only, total {len(data_hop3)} items.")

    outfile_hop4 = output_path.replace('.jsonl', '_hop4.jsonl')
    with open(outfile_hop4, 'w', encoding='utf-8') as f_out:
        for item in data_hop4:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Filtered data with hop4 only, total {len(data_hop4)} items.")

    logger.info(f"=== 任务完成 ===")
    logger.info(f"输入样本总数: {total_input}")
    logger.info(f"成功生成 multihop QA 数: {total_success}")


if __name__ == "__main__":
    INPUT_JSONL_PATH = ""
    OUTPUT_JSONL_PATH = ""
    run_multihop_parallel(
        input_path=INPUT_JSONL_PATH,
        output_path=OUTPUT_JSONL_PATH,
        max_workers=200,
        limit=5000,
    )
