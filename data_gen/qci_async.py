import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
import json
from prompts import IMAGE_QUALITY_CHECK_SYSTEM_PROMPT
import re
import base64
from openai import AsyncOpenAI

CONCURRENCY = 30  # tune based on your local server capacity

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = AsyncOpenAI(
    base_url="http://localhost:8995/v1",
    api_key="EMPTY"
)

BASE_DIR = '/data1/lokesh/v2_zip_openphish_usable'
SAVE_DIR = 'NEW_DIR_openphish_v2'


async def process_folder(folder_name: str, semaphore: asyncio.Semaphore):
    save_path = Path(f'{SAVE_DIR}/{folder_name}/image.json')
    if save_path.exists():
        return

    image_path = f'{BASE_DIR}/{folder_name}/screenshot.jpg'
    encoded = encode_image(image_path)

    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model="gemma",
                messages=[
                    {"role": "system", "content": IMAGE_QUALITY_CHECK_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                            {"type": "text", "text": "Classify this webpage screenshot."}
                        ]
                    }
                ],
                max_tokens=4096
            )
    except Exception as e:
        print(f"Error processing folder {folder_name}: {e}")
        return
    response_plain_text = response.choices[0].message.content
    answer = re.search(r'"verdict"(.*?)"confidence"', response_plain_text, re.DOTALL)
    verdict_block = answer.group(1).strip()
    value = re.search(r'"(.*?)"', verdict_block, re.DOTALL).group(1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(value, f)


async def main():
    # folder_list = sorted([f.name for f in Path(BASE_DIR).iterdir() if f.is_dir() and int(f.name)>44450 and not Path(f'{SAVE_DIR}/{f.name}/image.json').exists()], key=lambda x: int(x))
    # folder_list = json.load(open('shubho_tranco.json'))
    shubho_usable = json.load(open('shubho_usable.json'))
    folder_list = [key for key in json.load(open('retain_dict_may28.json')).values() if not key in shubho_usable]
    print(f"Total folders to process: {len(folder_list)}")
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_folder(folder_name, semaphore) for folder_name in folder_list]
    
    # tqdm.gather gives a progress bar over concurrent tasks
    await tqdm.gather(*tasks, total=len(tasks))


if __name__ == "__main__":
    asyncio.run(main())