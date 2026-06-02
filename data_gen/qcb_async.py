import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
import json
from prompts import IMAGE_QUALITY_CHECK_SYSTEM_PROMPT
import re
import base64
from openai import AsyncOpenAI
from preprocess_utils import SoM

CONCURRENCY1 = 15  # tune based on your local server capacity
CONCURRENCY2 = 30

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


client = AsyncOpenAI(
    base_url="http://localhost:8995/v1",
    api_key="EMPTY"
)

BASE_DIR = '/data1/lokesh/v2_zip_openphish_usable'
SAVE_DIR = 'box_usability_shubho'


async def check_usability(folder, child, semaphore: asyncio.Semaphore):    
    if Path(f'{SAVE_DIR}/{folder}/{child}.json').exists() or Path(f'bus_json/{folder}.json').exists():        
        return
    img_path = Path(f'{BASE_DIR}/{folder}/screenshots/{child}.jpg')

    async with semaphore:
        encoded = await asyncio.to_thread(encode_image, img_path)

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

    response_text = response.choices[0].message.content

    try:
        answer = re.search(r'"verdict"(.*?)"confidence"', response_text, re.DOTALL)
        verdict_block = answer.group(1).strip()
        verdict_match = re.search(r'"(.*?)"', verdict_block, re.DOTALL)
        Path(f'{SAVE_DIR}/{folder}').mkdir(parents=True, exist_ok=True)
        with open(f'{SAVE_DIR}/{folder}/{child}.json', 'w') as f:
            json.dump({"verdict": verdict_match.group(1)}, f)
    except Exception as e:
        with open(f'{SAVE_DIR}/{folder}/{child}.json', 'w') as f:
            json.dump({"verdict": "error", "error_message": str(e), "response_text": response_text}, f)


def _process_folder_sync(folder_name):
    """Blocking SoM work isolated here so it can run in a thread."""
    folder_path = Path(f'{BASE_DIR}/{folder_name}')

    sommer = SoM(
        folder_path / 'base_screenshots',
        folder_path / 'data.json',
        folder_path / 'base_screenshots/metadata.json',
    )
    active_boxes = sommer.get_active_boxes()
    return [(folder_name, child) for child in active_boxes]


async def process_folder(folder_name):
    # Run blocking SoM I/O in a thread to avoid blocking the event loop
    return await asyncio.to_thread(_process_folder_sync, folder_name)


async def main():
    # with open('tranco_usable.json') as f:
    #     folder_list = json.load(f)
    folder_list = json.load(open('retain_dict_may28.json')).values()
    folder_list = sorted([f for f in folder_list if Path(f'NEW_DIR_tranco/{f}/image.json').exists() and json.load(open(f'NEW_DIR_tranco/{f}/image.json')) == 'usable'], key=lambda x: int(x))

    semaphore = asyncio.Semaphore(CONCURRENCY1)

    folder_results_list = await tqdm.gather(
        *[process_folder(folder_name) for folder_name in folder_list],
        total=len(folder_list),
        desc="Extracting active boxes"
    )

    all_results = [item for sublist in folder_results_list for item in sublist]

    semaphore = asyncio.Semaphore(CONCURRENCY2)
    usability_tasks = [check_usability(folder, child, semaphore) if not Path(f'{SAVE_DIR}/{folder}/{child}.json').exists() else None for folder, child in all_results]
    usability_tasks = [task for task in usability_tasks if task is not None]
    usability_results = await tqdm.gather(
        *usability_tasks,
        total=len(usability_tasks),
        desc="Checking usability"
    )


if __name__ == "__main__":
    asyncio.run(main())