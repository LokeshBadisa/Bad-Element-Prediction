#Qwen 3.5 2B
# CUDA_VISIBLE_DEVICES=0 /data1/lokesh/miniconda3/envs/lf_vllm/bin/python /data1/lokesh/LlamaFactory/scripts/vllm_infer.py \
#   --model_name_or_path Qwen/Qwen3.5-2B \
#   --dataset beptest \
#   --template qwen3_5_nothink \
#   --cutoff_len 32000 \
#   --save_name results/qwen3_5_2b_predictions.jsonl \
#   --dataset_dir /data1/lokesh/LlamaFactory/data


# #Qwen/Qwen3.5-27B
# CUDA_VISIBLE_DEVICES=0,1 /data1/lokesh/miniconda3/envs/lf_vllm/bin/python /data1/lokesh/LlamaFactory/scripts/vllm_infer.py \
#   --model_name_or_path Qwen/Qwen3.5-27B \
#   --dataset beptest \
#   --template qwen3_5_nothink \
#   --cutoff_len 32000 \
#   --save_name results/qwen3_5_27b_predictions.jsonl \
#   --dataset_dir /data1/lokesh/LlamaFactory/data

# #Qwen/Qwen3.5-35B-A3B
# CUDA_VISIBLE_DEVICES=0,1 /data1/lokesh/miniconda3/envs/lf_vllm/bin/python /data1/lokesh/LlamaFactory/scripts/vllm_infer.py \
#   --model_name_or_path Qwen/Qwen3.5-35B-A3B \
#   --dataset beptest \
#   --template qwen3_5_nothink \
#   --cutoff_len 32000 \
#   --save_name results/qwen3_5_35ba3b_predictions.jsonl \
#   --dataset_dir /data1/lokesh/LlamaFactory/data  

# #Qwen/Qwen3.6-27B
# CUDA_VISIBLE_DEVICES=0,1 /data1/lokesh/miniconda3/envs/lf_vllm/bin/python /data1/lokesh/LlamaFactory/scripts/vllm_infer.py \
#   --model_name_or_path Qwen/Qwen3.6-27B \
#   --dataset beptest \
#   --template qwen3_6 \
#   --cutoff_len 32000 \
#   --save_name results/qwen3_6_27b_predictions.jsonl \
#   --dataset_dir /data1/lokesh/LlamaFactory/data

# #Qwen/Qwen3.6-35B-A3B
# CUDA_VISIBLE_DEVICES=0,1 /data1/lokesh/miniconda3/envs/lf_vllm/bin/python /data1/lokesh/LlamaFactory/scripts/vllm_infer.py \
#   --model_name_or_path Qwen/Qwen3.6-35B-A3B \
#   --dataset beptest \
#   --template qwen3_6 \
#   --cutoff_len 32000 \
#   --save_name results/qwen3_6_35ba3b_predictions.jsonl \
#   --dataset_dir /data1/lokesh/LlamaFactory/data

# #google/gemma-4-E2B-it
# CUDA_VISIBLE_DEVICES=0 /data1/lokesh/miniconda3/envs/gemma_vllm/bin/python baseline.py \
#   --model_name_or_path google/gemma-4-E2B-it \
#   --save_name results/gemma4_e2b_predictions.jsonl \
#   --tp_size 1

#google/gemma-4-31B-it
CUDA_VISIBLE_DEVICES=0,1 /data1/lokesh/miniconda3/envs/gemma_vllm/bin/python baseline.py \
  --model_name_or_path google/gemma-4-31B-it \
  --save_name results/gemma4_31b_predictions.jsonl

#google/gemma-4-26B-A4B-it
CUDA_VISIBLE_DEVICES=0,1 /data1/lokesh/miniconda3/envs/gemma_vllm/bin/python baseline.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --save_name results/gemma4_26ba4b_predictions.jsonl

#google/gemma-4-E4B-it
CUDA_VISIBLE_DEVICES=0,1 /data1/lokesh/miniconda3/envs/gemma_vllm/bin/python baseline.py \
  --model_name_or_path google/gemma-4-E4B-it \
  --save_name results/gemma4_e4b_predictions.jsonl
  