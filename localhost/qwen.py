import os
os.system('CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-8B --served-model-name openai/Qwen3-8B --host 0.0.0.0 --port 8000 --enforce-eager')