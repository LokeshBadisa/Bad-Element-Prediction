import asyncio
import copy
import json
from collections import defaultdict
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import re
import json
from tqdm import tqdm
from pathlib import Path
from preprocess_utils import SoM
from prompts import SYSTEM_PROMPT
import copy
from collections import defaultdict
from labelling_utils import extract_all_features

def extract_labels(content):
    json_blocks = re.findall(r'```json(.*?)```', content, re.DOTALL)
        
    for match in json_blocks:
        cleaned = re.sub(r'\s+', ' ', match).strip()
        D = json.loads(cleaned)
        if 'final_label' in D:
            return D['final_label']
    return None


OPENPHISH_BASE_DIR = '/data1/lokesh/v2_zip_openphish_usable'
OPENPHISH_REASONING_FOLDER = 'GLR_json'
OPENPHISH_CONFLICT_RES_FOLDER = 'cres_outputs'
OPENPHISH_BOX_CROP_FOLDER = 'openphish_box_crops'

TRANCO_BASE_DIR = '/data1/lokesh/tranco_data/bep/data_gen/data'
TRANCO_BOX_CROP_FOLDER = 'tranco_box_crops'

OPENPHISH_FINAL_FOLDER = 'openphish_complete_data_sharegpt'
TRANCO_FINAL_FOLDER = 'tranco_complete_data_sharegpt'
SHAREGPT_OUTPUT_FILE = 'trainset_sharegpt_data_may23.json'


async def process_folder(folder, TYPE, REASONING_FOLDER, CONFLICT_RES_FOLDER,
                         BOX_CROP_FOLDER, FINAL_FOLDER, semaphore):
    async with semaphore:
        loop = asyncio.get_event_loop()

        # ── JSON loads (offloaded to thread pool; they're blocking I/O) ──────
        jsondata = await loop.run_in_executor(
            None, lambda: json.load(open(folder / 'data.json'))
        )

        if TYPE == 'malicious':
            reasoning_path = Path(f'{REASONING_FOLDER}/{folder.name}.json')
            if not reasoning_path.exists():
                return None, folder.name          # signals removal
            labelsdata = await loop.run_in_executor(
                None, lambda: json.load(open(reasoning_path))
            )
        else:
            labelsdata = defaultdict(lambda: ['benign'])

        scrolldata = await loop.run_in_executor(
            None,
            lambda: json.load(
                open(folder / 'base_screenshots/metadata.json')
            )['scroll_steps']
        )

        # ── Prepare base message skeleton ────────────────────────────────────
        temp_dict_base = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "images": []
        }

        Path(f'{FINAL_FOLDER}/{folder.name}').mkdir(parents=True, exist_ok=True)

        # SoM does CPU + disk work — run in executor so it doesn't block loop
        sommer = await loop.run_in_executor(
            None,
            lambda: SoM(
                folder / 'base_screenshots',
                folder / 'data.json',
                folder / 'base_screenshots/metadata.json',
                process_all_boxes=True,
                crop_boxes=True,
                crop_location=f'{BOX_CROP_FOLDER}/{folder.name}'
            )
        )
        await loop.run_in_executor(
            None, lambda: sommer.save(f'{FINAL_FOLDER}/{folder.name}')
        )

        # ── Build label_dict ─────────────────────────────────────────────────
        label_dict = {}
        for i in range(len(scrolldata) + 1):
            boxes_in_image = sommer.boxes_in_image.get(i, [])
            for key in sorted(boxes_in_image):
                try:
                    if TYPE != 'malicious':
                        label_dict[key] = 'benign'
                        continue

                    conflict_path = Path(
                        f'{CONFLICT_RES_FOLDER}/{folder.name}_{key}.json'
                    )
                    if conflict_path.exists():
                        extracted = extract_labels(
                            await loop.run_in_executor(
                                None, lambda p=conflict_path: open(p).read()
                            )
                        )
                    else:
                        extracted = extract_labels(labelsdata[f'{i}_{key}'])

                    if extracted is None or 'obfuscated' in extracted:
                        sommer.boxes_in_image[i].remove(key)
                    else:
                        label_dict[key] = extracted
                except Exception:
                    sommer.boxes_in_image[i].remove(key)

        # ── Build ShareGPT records ───────────────────────────────────────────
        records = []
        extracted_feats = extract_all_features(jsondata['base'])

        for i in range(len(scrolldata) + 1):
            boxes_in_image = sommer.boxes_in_image.get(i, [])
            if not boxes_in_image:
                continue

            temp_dict = copy.deepcopy(temp_dict_base)

            color_dict      = {box: sommer.color_list[f'{i}_{box}'] for box in boxes_in_image}
            image_width     = sommer.outputs[i].width
            image_height    = sommer.outputs[i].height
            center_ys       = {box: jsondata[box]["y"] - sum(scrolldata[:i]) + jsondata[box]['height'] / 2 for box in boxes_in_image}
            center_xs       = {box: jsondata[box]["x"] + jsondata[box]['width'] / 2                       for box in boxes_in_image}
            paired_centers  = {box: (center_xs[box], center_ys[box])                                      for box in boxes_in_image}
            norm_xs         = {box: center_xs[box] / image_width                                          for box in boxes_in_image}
            norm_ys         = {box: center_ys[box] / image_height                                         for box in boxes_in_image}
            paired_norm_centers = {box: (norm_xs[box], norm_ys[box])                                      for box in boxes_in_image}

            image_string = "<image> " * (len(boxes_in_image) + 1)

            temp_dict["messages"].append({
                "role": "human",
                "content": (
                    f"{image_string} Classify the elements in given image. "
                    f"URL of this page is {jsondata['base']}. "
                    f"Boxes in the image are: {', '.join(boxes_in_image)}. "
                    f"Each box is highlighted with these colors: {color_dict}. "
                    f"Centers of the boxes are: {paired_centers}. "
                    f"Image size: {(image_width, image_height)}. "
                    f"Normalized centers of the boxes are: {paired_norm_centers}. "
                    f"Extracted features of the URL are: {extracted_feats}. "
                )
            })
            temp_dict["messages"].append({
                "role": "gpt",
                "content": (
                    '```json{'
                    + ', '.join(f'{k}:{label_dict[k]}' for k in sorted(boxes_in_image))
                    + '}```'
                )
            })
            temp_dict["images"].append(
                str(Path(f'{FINAL_FOLDER}/{folder.name}/{i}.jpg').absolute())
            )
            for box in boxes_in_image:
                temp_dict["images"].append(
                    str(Path(f'{BOX_CROP_FOLDER}/{folder.name}/{box}.jpg').absolute())
                )
            
            records.append(temp_dict)

        return records, None   # None = no removal needed


async def create_data(folder_list, TYPE, BASE_DIR,
                      REASONING_FOLDER=None, CONFLICT_RES_FOLDER=None,
                      BOX_CROP_FOLDER=None, FINAL_FOLDER=None,
                      max_concurrency: int = 8):
    """
    max_concurrency controls how many folders are processed simultaneously.
    Tune this based on available RAM and disk I/O capacity.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    folders   = [Path(f'{BASE_DIR}/{f}') for f in folder_list]

    tasks = [
        process_folder(
            folder, TYPE, REASONING_FOLDER, CONFLICT_RES_FOLDER,
            BOX_CROP_FOLDER, FINAL_FOLDER, semaphore
        )
        for folder in folders
    ]

    sharegpt    = []
    remove_list = []

    results = await tqdm_asyncio.gather(*tasks, desc="Processing folders")

    for result in results:
        records, removed = result
        if removed:
            remove_list.append(removed)
        elif records:
            sharegpt.extend(records)

    return sharegpt

async def main():
    openphish_folder_list = open('train_set_may23_shubho.txt').read().splitlines()
    tranco_folder_list = open('train_set_may23_tranco.txt').read().splitlines()
    sharegpt_openphish =  await create_data(openphish_folder_list, 'malicious', OPENPHISH_BASE_DIR, OPENPHISH_REASONING_FOLDER, OPENPHISH_CONFLICT_RES_FOLDER, OPENPHISH_BOX_CROP_FOLDER, OPENPHISH_FINAL_FOLDER)
    sharegpt_tranco = await create_data(tranco_folder_list, 'benign', TRANCO_BASE_DIR, BOX_CROP_FOLDER=TRANCO_BOX_CROP_FOLDER, FINAL_FOLDER=TRANCO_FINAL_FOLDER)
    

    final_sharegpt = sharegpt_openphish + sharegpt_tranco
    with open(SHAREGPT_OUTPUT_FILE, "w") as f:
        json.dump(final_sharegpt, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())