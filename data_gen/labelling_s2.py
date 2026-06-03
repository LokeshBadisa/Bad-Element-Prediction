import asyncio
import pandas as pd
import base64
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from pathlib import Path
from prompts import LABEL_GENERATION_SYSTEM_PROMPT

CONCURRENCY = 30
SAVE_DIR = 'Gemma_Long_Run_new'

client = AsyncOpenAI(
    base_url="http://localhost:8995/v1",
    api_key="EMPTY"
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

async def process_row(semaphore, row):
    if Path(f'{SAVE_DIR}/{row["id"]}.txt').exists():
        return  # Skip if already processed
    async with semaphore:
        prompt_text = (
            f"Box is highlighted with {row['color']} color. "
            f"Center of the highlighted box is {row['center_x']} and {row['center_y']} in raw pixel space. "
            f"Image Size is {row['image_width']} and {row['image_height']}. "
            f"Normalized Center of the highlighted box is {row['normalized_center_x']} and {row['normalized_center_y']}. "
            f"URL of first image is {row['url1']}. URL of second image is {row['url2']}. "
            f"Derived features from URLs of both images are {row['derived_url_features']}. "
            f"did_anything_download = {row['did_anything_download']}"
        )

        img1, img2, img3 = await asyncio.gather(
            asyncio.to_thread(encode_image, row['img_url1']),
            asyncio.to_thread(encode_image, row['img_url2']),
            asyncio.to_thread(encode_image, row['img_url3']),
        )

        response = await client.chat.completions.create(
            model="gemma",
            messages=[
                {"role": "system", "content": LABEL_GENERATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img3}"}},
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

        with open(f'{SAVE_DIR}/{row["id"]}.txt', 'w') as f:
            f.write(response_text)

async def main():
    Path(SAVE_DIR).mkdir(exist_ok=True)
    df = pd.read_csv('reader_temp.csv')
    # with open('gemma_long_run_missing.txt') as f:
    #     missing = set(f.read().splitlines())
    # df = df[df['id'].isin(missing)]    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_row(semaphore, df.iloc[i]) for i in range(len(df))]
    await tqdm.gather(*tasks, total=len(tasks), desc="Labelling")

if __name__ == "__main__":
    asyncio.run(main())
