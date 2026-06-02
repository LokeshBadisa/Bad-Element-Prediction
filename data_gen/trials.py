import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from PIL import Image
from preprocess_utils import *
import json


BASE_DIR = '/data1/lokesh/tranco_data/bep/data_gen/data'
train_set = [line.strip() for line in open('train_set_may23_tranco.txt')]

async def process_folder(folder, semaphore):
    async with semaphore:
        try:
            sommer = await asyncio.to_thread(
                SoM,
                f'{BASE_DIR}/{folder}/base_screenshots',
                Path(f'{BASE_DIR}/{folder}/data.json'),
                f'{BASE_DIR}/{folder}/base_screenshots/metadata.json'
            )
            return folder, sommer
        except Image.DecompressionBombWarning:
            print(f"Skipping folder {folder} due to image size issues.")
            return folder, None

async def main():
    semaphore = asyncio.Semaphore(16)  # tune to your CPU/IO bottleneck

    tasks = [process_folder(folder, semaphore) for folder in train_set]
    results = await tqdm.gather(*tasks, desc="Processing folders")

    return {folder: som for folder, som in results if som is not None}

if __name__ == "__main__":
    L = asyncio.run(main())    