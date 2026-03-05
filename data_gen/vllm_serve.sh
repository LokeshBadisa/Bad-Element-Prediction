CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --served-model-name qwen \
    --model "Qwen/Qwen3-VL-8B-Instruct" \
    --limit-mm-per-prompt '{"image": 4,"video": 0}'\
    -tp 2 --gpu-memory-utilization 0.9 --max-model-len 16384 --port 9000\
    --async-scheduling