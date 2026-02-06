# AgenticRAGTracer: A Hop-Aware Benchmark for Diagnosing Multi-Step Retrieval Reasoning in Agentic RAG

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/YqjMartin/AgenticRAGTracer)

---

## 📁 Repository Structure

```bash
AgenticRAGTrace
├── multihop_pipeline.py      # Core multi-hop generation pipeline
├── multihop_run.py           # Parallel runner for multi-hop QA generation
├── multihop_prompt.yaml      # The prompts used in the multi-hop generation pipeline
├── evaluation.py             # Agentic RAG evaluation script
├── retriever_serving.py      # Retriever service
├── retriever_config.yaml     # Retriever configuration
└── README.md
```

---

## 🔎 Retriever Service

First, RAG serving needs to be started. Below are some of the configurations that need to be made in the code.

``` bash
# retriever_config.yaml
gpu_id: ""
retrieval_method: "e5"
retrieval_model_path: "e5-base-v2"
index_path: "e5_flat_inner.index" 
faiss_gpu: False
corpus_path: "wiki18_100w.jsonl"  
```

``` bash
# retriever_serving.py
python retriever_serving.py \
  --config retriever_config.yaml \
  --port 8000
```

---

## 🔄 Multi-hop Data Generation

You can use the code of multihop_run.py and multihop_pipeline.py to build the Multihop QA of AgenticRAG. Below are some of the configurations you need to make in the code.

``` bash
# multihop_pipeline.py
API_URL = ""                        # The URL address of the LLM API you are using
API_KEY = ""                        # The API key
DEFAULT_MODEL = "gpt-4o-mini"       # The model you want to use for generation
```

``` bash    
# multihop_run.py
python multihop_run.py
```

---

## 📊 Evaluation

You can use the code in evaluation.py to evaluate the LLM that you want to assess. Below are some of the configurations you need to make.