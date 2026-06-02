import asyncio
import ast
from pathlib import Path
from tqdm.asyncio import tqdm
import json
import pandas as pd
import re
import base64
from openai import AsyncOpenAI
from prompts import CONFLICT_RESOLUTION_SYSTEM_PROMPT

CONCURRENCY = 30
SAVE_DIR = 'cres_outputs'

client = AsyncOpenAI(
    base_url="http://localhost:8995/v1",
    api_key="EMPTY"
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

async def process_row(semaphore, row):
    try:
        async with semaphore:
            _id = row['id']
            if Path(f'{SAVE_DIR}/{_id}.txt').exists():
                return  # Skip if already processed
            boxes = ast.literal_eval(row['boxes'])
            coords = ast.literal_eval(row['coords'])
            n_coords = ast.literal_eval(row['n_coords'])
            img_sizes = ast.literal_eval(row['img_sizes'])
            base_url = row['base_url']
            box_url = row['box_url']
            curr_reasoning = ast.literal_eval(row['reasoning'])
            did_anything_download = ast.literal_eval(str(row['did_anything_download']))
            colors = ast.literal_eval(row['colors'])
            derived_features = ast.literal_eval(row['derived_features'])
            init_images = ast.literal_eval(row['init_images'])
            img_url2 = row['img_url2']
            img_url3 = row['img_url3']

            reasoning_by_index = {box: curr_reasoning[box] for box in boxes}
            prompt_text = (
                f"Boxes are highlighted with {[colors[box] for box in boxes]} colors in {list(range(1, len(boxes)+1))} images respectively. "
                f"Center of the highlighted boxes are {coords} in raw pixel space. Image Sizes are {img_sizes}. "
                f"Normalized Center of the highlighted box is {n_coords}. "
                f"Image ids are {boxes}. "
                f"URL of {list(range(1, len(boxes)+1))} images are {base_url}. URL of second image is {box_url}. "
                f"Derived features from URLs of both images are {derived_features}. "
                f"did_anything_download = {did_anything_download}. "
                f"Reasoning for each box is {reasoning_by_index}. "
            )

            encoded_init = await asyncio.gather(
                *[asyncio.to_thread(encode_image, img) for img in init_images]
            )
            enc2, enc3 = await asyncio.gather(
                asyncio.to_thread(encode_image, img_url2),
                asyncio.to_thread(encode_image, img_url3),
            )

            user_dict = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}}
                for enc in encoded_init
            ]

            response = await client.chat.completions.create(
                model="gemma",
                messages=[
                    {"role": "system", "content": CONFLICT_RESOLUTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            *user_dict,
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc2}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc3}"}},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ],
                max_tokens=8192
            )

            message = response.choices[0].message
            response_text = message.content
            if hasattr(message, "reasoning") and message.reasoning:
                response_text += f"\n\n[Reasoning]: {message.reasoning}"

            with open(f'{SAVE_DIR}/{_id}.txt', 'w') as f:
                f.write(response_text)
    except Exception as e:
        print(f"Error processing row with id {_id}: {e}")


async def main():
    Path(SAVE_DIR).mkdir(exist_ok=True)
    df = pd.read_csv('conflict_resolution_data.csv')
    done = {p.stem for p in Path(SAVE_DIR).iterdir()}
    df = df[~df['id'].isin(done)]
    #take only the where id is 816_5
    df = df[df['id'] == '816_5']
    print(f"Processing {len(df)} rows ({len(done)} already done)")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_row(semaphore, df.iloc[i]) for i in range(len(df))]
    await tqdm.gather(*tasks, total=len(tasks), desc="Conflict Resolution")

if __name__ == "__main__":
    asyncio.run(main())
