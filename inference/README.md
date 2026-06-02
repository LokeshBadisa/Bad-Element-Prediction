```
pip install vllm --no-build-isolation
llamafactory-cli export examples/merge_lora/qwen3_vl_lora_sft.yaml 
cd LlamaFactory
CUDA_VISIBLE_DEVICES=3 /data1/lokesh/miniconda3/envs/lf_vllm/bin/python scripts/vllm_infer.py \
  --model_name_or_path saves/qwen3_5_vl_sft_merged/ \
  --dataset beptest \
  --template qwen3_5_nothink \
  --cutoff_len 32000
cd inference
python3 test.py
```