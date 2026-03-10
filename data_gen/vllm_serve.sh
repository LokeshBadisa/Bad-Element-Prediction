CUDA_VISIBLE_DEVICES=2,1 python -m vllm.entrypoints.openai.api_server \
    --served-model-name qwen \
    --model "Qwen/Qwen3.5-35B-A3B" \
    --limit-mm-per-prompt '{"image": 4,"video": 0}'\
    -tp 2 --gpu-memory-utilization 0.9 --max-model-len 16384 --port 9013\
    --async-scheduling