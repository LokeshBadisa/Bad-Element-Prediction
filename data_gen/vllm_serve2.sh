CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3.5-27B --port 8996\
 --served-model-name qwen\
 --tensor-parallel-size 2 --max-model-len 262144\
 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder