import asyncio
import base64
import json
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from pathlib import Path
from prompts import WEBPAGE_LABEL_COT_SYSTEM_PROMPT
from labelling_utils import extract_all_features

CONCURRENCY = 30
# SAVE_DIR = 'Tranco_WL_CoT'
# BASE_DIR = '/data1/lokesh/tranco_data/bep/data_gen/data'
SAVE_DIR = 'Openphish_WL_CoT'
BASE_DIR = '/data1/lokesh/v2_zip_openphish_usable'

client = AsyncOpenAI(
    base_url="http://localhost:8995/v1",
    api_key="EMPTY"
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

async def process_row(semaphore, folder_name):
    if Path(f'{SAVE_DIR}/{folder_name}.txt').exists():
        return  # Skip if already processed

    url = json.load(open(f'{BASE_DIR}/{folder_name}/data.json'))['base']
    async with semaphore:
        prompt_text = (
            f"URL of the webpage is {url}."
            f"Derived features from the URL are {extract_all_features(url)}."
            f"Groundtruth label: malicious"
        )

        img1 = encode_image(f'{BASE_DIR}/{folder_name}/screenshot.jpg')

        response = await client.chat.completions.create(
            model="gemma",
            messages=[
                {"role": "system", "content": WEBPAGE_LABEL_COT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1}"}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            max_tokens=10240
        )

        message = response.choices[0].message
        response_text = message.content
        if hasattr(message, "reasoning") and message.reasoning:
            response_text += f"\n\n[Reasoning]: {message.reasoning}"

        with open(f'{SAVE_DIR}/{folder_name}.txt', 'w') as f:
            f.write(response_text)

async def main():
    Path(SAVE_DIR).mkdir(exist_ok=True)
    # json_data = json.load(open('tranco_usable.json'))
    json_data = json.load(open('retain_dict_may28.json')).values()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_row(semaphore, folder_name) for folder_name in json_data]
    await tqdm.gather(*tasks, total=len(tasks), desc="Labelling")

if __name__ == "__main__":
    asyncio.run(main())
