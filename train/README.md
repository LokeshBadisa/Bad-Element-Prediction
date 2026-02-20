Install [LLaMaFactory](https://github.com/hiyouga/LlamaFactory?tab=readme-ov-file#installation). While installing, force it to install torch 2.8. Note that torch 2.9's Conv3d is slow and this will take more time to train/inference.

Use 7901b2f32e8e1f3261f3cff882773490b2a3d7a5 commit of LLaMaFactory only for installation. Any other version doesn't work. After that do `pip install torch==2.9.1 torchvision --no-build-isolation`

Add the below entry to `LLaMaFactory/data/dataset_info.json`
```json
"bep": {
    "file_name": "/data1/lokesh/blp/sharegpt_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "system_tag": "system"
    }
  }
```

Create the below config file for training at `LlamaFactory/examples/train_lora/qwen3_lora_sft_ds3.yaml`
```yaml
### model
model_name_or_path: Qwen/Qwen3-VL-2B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
deepspeed: examples/deepspeed/ds_z2_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: bep
template: qwen3_vl_nothink
cutoff_len: 16384
max_samples: 1000
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3vl-2b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: constant
warmup_ratio: 0
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: false

### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
```

Now,
```bash
cd LLaMaFactory
llamafactory-cli train examples/train_lora/qwen3_lora_sft_ds3.yaml
```