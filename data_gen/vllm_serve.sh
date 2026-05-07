# CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
#     --served-model-name qwen \
#     --model "Qwen/Qwen3-VL-30B-A3B-Thinking" \
#     --limit-mm-per-prompt '{"image": 4,"video": 0}'\
#     -tp 2 --gpu-memory-utilization 0.9 --max-model-len 16384 --port 9013\
#     --async-scheduling

# CUDA_VISIBLE_DEVICES=2,3 vllm serve Qwen/Qwen3.5-27B --port 9013\
#  --served-model-name qwen
#  --tensor-parallel-size 2 --max-model-len 262144\
#  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

CUDA_VISIBLE_DEVICES=0,1 vllm serve google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --port 8995 --served-model-name gemma \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template tool_chat_template_gemma4.jinja \
  --default-chat-template-kwargs '{"enable_thinking": true}'