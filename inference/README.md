```
pip install vllm --no-build-isolation
llamafactory-cli export examples/merge_lora/qwen3_vl_lora_sft.yaml 
cd LlamaFactory
python scripts/vllm_infer.py --model_name_or_path output_wourl/qwen3_vl_lora_sft/ --dataset beptest_wourl --template qwen3_vl_nothink
cd inference
python3 test.py
```