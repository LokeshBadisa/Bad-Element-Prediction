import argparse
import json

from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", required=True)
parser.add_argument("--save_name", required=True)
parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallel size (default: 2; use 1 for single-GPU)")
parser.add_argument("--max_model_len", type=int, default=32000)
parser.add_argument("--max_new_tokens", type=int, default=2048)
args = parser.parse_args()

DATASET = "/data1/lokesh/bep/data_gen/testset_sharegpt_data_may23.json"

with open(DATASET) as f:
    dataset = json.load(f)

ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def build_chat_messages(sample: dict) -> list[dict]:
    """Convert a ShareGPT sample to vLLM multimodal chat format."""
    images = sample["images"]
    img_idx = 0
    result = []

    for msg in sample["messages"]:
        role = ROLE_MAP.get(msg["role"], msg["role"])
        text = msg["content"]

        if "<image>" in text:
            parts = text.split("<image>")
            content = []
            for i, part in enumerate(parts):
                if i > 0 and img_idx < len(images):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{images[img_idx]}"},
                    })
                    img_idx += 1
                stripped = part.strip()
                if stripped:
                    content.append({"type": "text", "text": stripped})
            result.append({"role": role, "content": content})
        else:
            result.append({"role": role, "content": text})

    # Drop the last assistant turn — that's what we predict
    if result and result[-1]["role"] == "assistant":
        result = result[:-1]

    return result


llm = LLM(
    model=args.model_name_or_path,
    tensor_parallel_size=args.tp_size,
    max_model_len=args.max_model_len,
    limit_mm_per_prompt={"image": 100},
    allowed_local_media_path="/data1/lokesh/bep/data_gen",
    disable_log_stats=True,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=args.max_new_tokens,
)

all_messages = [build_chat_messages(sample) for sample in dataset]
outputs = llm.chat(all_messages, sampling_params=sampling_params)

import os
os.makedirs(os.path.dirname(args.save_name) if os.path.dirname(args.save_name) else ".", exist_ok=True)

with open(args.save_name, "w") as f:
    for sample, out in zip(dataset, outputs):
        f.write(json.dumps({
            "predict": out.outputs[0].text,
            "label": sample["messages"][-1]["content"],
            "images": sample["images"],
        }, ensure_ascii=False) + "\n")

print(f"Saved {len(outputs)} predictions to {args.save_name}")
