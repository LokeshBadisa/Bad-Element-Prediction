from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
import json
import shutil
import base64
import asyncio
from labelling_utils import *
from prompts import LABEL_GENERATION_SYSTEM_PROMPT
from preprocess_utils import SoM
from openai import AsyncOpenAI

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


client = AsyncOpenAI(
    base_url="http://localhost:8996/v1",
    api_key="EMPTY"
)

MAX_CONCURRENT = 8  # tune to your local server's capacity
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def call_model(img_num, box, sommer, json_data, scroll_info, folder_path: Path):
    modified_y = json_data[box]["y"] - sum(scroll_info[:img_num])
    url1 = json_data['base']
    url2 = json_data[box]['url']

    prompt_text = (
        f"Box is highlighted with {sommer.color_list[f'{img_num}_{box}']} color. "
        f"Center of the highlighted box is {json_data[box]['x'] + json_data[box]['width'] / 2} and "
        f"{modified_y + json_data[box]['height'] / 2} in raw pixel space. "
        f"Image Size is {sommer.outputs[img_num].width} and {sommer.outputs[img_num].height}. "
        f"Normalized Center of the highlighted box is "
        f"{(json_data[box]['x'] + json_data[box]['width'] / 2) / sommer.outputs[img_num].width} and "
        f"{(modified_y + json_data[box]['height'] / 2) / sommer.outputs[img_num].height}. "
        f"URL of first image is {url1}. URL of second image is {url2}. "
        f"Derived features from URLs of both images are {deriveUrlFeatures(url1, url2)}."
        f"did_anything_download = False"
    )

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'temp_each_box/{img_num}/{box}.jpg')}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'{folder_path}/screenshots/{box}.jpg')}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'./box_crops/{folder_path.name}/{box}.jpg')}"}},
        {"type": "text", "text": prompt_text},
    ]

    async with semaphore:
        response = await client.chat.completions.create(
            model="qwen",
            messages=[
                {"role": "system", "content": LABEL_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=5120,
        )

    message = response.choices[0].message
    response_text = message.content
    if hasattr(message, "reasoning") and message.reasoning is not None:
        try:
            response_text = "[Reasoning]:\n" + message.reasoning + "\n[Answer]:\n" + response_text
        except Exception as e:
            pass

    return f"{img_num}_{box}", response_text


async def main():
    folders = sorted(
        [f for f in Path('/data1/lokesh/combineddata').iterdir()],
        key=lambda x: int(x.name)
    )[:20]

    for folder in tqdm(folders, desc="Folders"):
        if Path(f'qwen3.5_vlm1_reasoning_wobf/{folder.name}.json').exists():            
            continue
        json_data = json.load(open(f'{folder}/data.json'))
        scroll_info = json.load(open(f'{folder}/base_screenshots/metadata.json'))['scroll_steps']

        sommer = SoM(
            f'{folder}/base_screenshots',
            Path(f'{folder}/data.json'),
            f'{folder}/base_screenshots/metadata.json',
            process_each_box=True,
            process_eb_folder='./temp_each_box',
            crop_boxes=True,
            crop_location=f'./box_crops/{folder.name}'
        )

        tasks = []
        for img_num, boxes_list in sommer.boxes_in_image.items():
            for box in boxes_list:
                if "error: " in json_data[box]["url"]:
                    continue
                tasks.append(call_model(img_num, box, sommer, json_data, scroll_info, folder))

        results = await async_tqdm.gather(*tasks, desc=f"  Boxes in folder {folder.name}")
        answers_dict = dict(results)

        shutil.rmtree('./temp_each_box')
        Path('qwen3.5_vlm1_reasoning_wobf').mkdir(parents=True, exist_ok=True)
        with open(f'qwen3.5_vlm1_reasoning_wobf/{folder.name}.json', 'w') as f:
            json.dump(answers_dict, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())