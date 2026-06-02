import asyncio
import copy
import json
from collections import defaultdict
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import re

from preprocess_utils import SoM
from labelling_utils import extract_all_features

LABEL_MAP = {'malicious': 1, 'benign': 0}

OD_SYSTEM_PROMPT = (
    "You are an object detection model. "
    "Given a webpage screenshot, output a JSON list of detected elements. "
    "Each entry must contain: box_id, bbox as [x1, y1, x2, y2] normalized to [0,1], "
    "and label where 1=malicious and 0=benign."
)

OPENPHISH_BASE_DIR = '/data1/lokesh/v2_zip_openphish_usable'
OPENPHISH_REASONING_FOLDER = 'GLR_json'
OPENPHISH_CONFLICT_RES_FOLDER = 'cres_outputs'
OPENPHISH_BOX_CROP_FOLDER = 'openphish_box_crops'

TRANCO_BASE_DIR = '/data1/lokesh/tranco_data/bep/data_gen/data'
TRANCO_BOX_CROP_FOLDER = 'tranco_box_crops'

OPENPHISH_FINAL_FOLDER = 'openphish_complete_data_od'
TRANCO_FINAL_FOLDER = 'tranco_complete_data_od'
OD_OUTPUT_FILE = 'trainset_od_data_may23.json'


def extract_labels(content):
    json_blocks = re.findall(r'```json(.*?)```', content, re.DOTALL)
    for match in json_blocks:
        cleaned = re.sub(r'\s+', ' ', match).strip()
        D = json.loads(cleaned)
        if 'final_label' in D:
            return D['final_label']
    return None


async def process_folder(folder, TYPE, REASONING_FOLDER, CONFLICT_RES_FOLDER,
                         BOX_CROP_FOLDER, FINAL_FOLDER, semaphore):
    async with semaphore:
        loop = asyncio.get_event_loop()

        jsondata = await loop.run_in_executor(
            None, lambda: json.load(open(folder / 'data.json'))
        )

        if TYPE == 'malicious':
            reasoning_path = Path(f'{REASONING_FOLDER}/{folder.name}.json')
            if not reasoning_path.exists():
                return None, folder.name
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

        temp_dict_base = {
            "messages": [{"role": "system", "content": OD_SYSTEM_PROMPT}],
            "images": []
        }

        Path(f'{FINAL_FOLDER}/{folder.name}').mkdir(parents=True, exist_ok=True)

        sommer = await loop.run_in_executor(
            None,
            lambda: SoM(
                folder / 'base_screenshots',
                folder / 'data.json',
                folder / 'base_screenshots/metadata.json',
                process_all_boxes=True,
                crop_boxes=False,
                crop_location=f'{BOX_CROP_FOLDER}/{folder.name}'
            )
        )
        await loop.run_in_executor(
            None, lambda: sommer.save(f'{FINAL_FOLDER}/{folder.name}')
        )

        # Build label_dict with numeric OD labels
        label_dict = {}
        for i in range(len(scrolldata) + 1):
            boxes_in_image = sommer.boxes_in_image.get(i, [])
            for key in sorted(boxes_in_image):
                try:
                    if TYPE != 'malicious':
                        label_dict[key] = LABEL_MAP['benign']
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
                        label_dict[key] = LABEL_MAP.get(extracted, LABEL_MAP['benign'])
                except Exception:
                    sommer.boxes_in_image[i].remove(key)

        records = []
        extracted_feats = extract_all_features(jsondata['base'])

        for i in range(len(scrolldata) + 1):
            boxes_in_image = sommer.boxes_in_image.get(i, [])
            if not boxes_in_image:
                continue

            temp_dict = copy.deepcopy(temp_dict_base)

            image_width  = sommer.outputs[i].width
            image_height = sommer.outputs[i].height
            scroll_offset = sum(scrolldata[:i])

            # Build normalized [x1, y1, x2, y2] bounding boxes
            annotations = []
            for box in sorted(boxes_in_image):
                bx = jsondata[box]["x"]
                by = jsondata[box]["y"] - scroll_offset
                bw = jsondata[box]["width"]
                bh = jsondata[box]["height"]

                x1 = max(0.0, bx / image_width)
                y1 = max(0.0, by / image_height)
                x2 = min(1.0, (bx + bw) / image_width)
                y2 = min(1.0, (by + bh) / image_height)

                annotations.append({
                    "box_id": box,
                    "bbox": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
                    "label": label_dict[box]
                })

            temp_dict["messages"].append({
                "role": "human",
                "content": (
                    f"<image> Detect and classify all UI elements in this webpage screenshot. "
                    f"URL: {jsondata['base']}. "
                    f"Image size: {image_width}x{image_height}. "
                    f"URL features: {extracted_feats}."
                )
            })
            temp_dict["messages"].append({
                "role": "gpt",
                "content": "```json\n" + json.dumps(annotations, separators=(',', ':')) + "\n```"
            })
            temp_dict["images"].append(
                str(Path(f'{FINAL_FOLDER}/{folder.name}/{i}.jpg').absolute())
            )

            records.append(temp_dict)

        return records, None


async def create_data(folder_list, TYPE, BASE_DIR,
                      REASONING_FOLDER=None, CONFLICT_RES_FOLDER=None,
                      BOX_CROP_FOLDER=None, FINAL_FOLDER=None,
                      max_concurrency: int = 8):
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
    tranco_folder_list    = open('train_set_may23_tranco.txt').read().splitlines()

    sharegpt_openphish = await create_data(
        openphish_folder_list, 'malicious', OPENPHISH_BASE_DIR,
        OPENPHISH_REASONING_FOLDER, OPENPHISH_CONFLICT_RES_FOLDER,
        OPENPHISH_BOX_CROP_FOLDER, OPENPHISH_FINAL_FOLDER
    )
    sharegpt_tranco = await create_data(
        tranco_folder_list, 'benign', TRANCO_BASE_DIR,
        BOX_CROP_FOLDER=TRANCO_BOX_CROP_FOLDER, FINAL_FOLDER=TRANCO_FINAL_FOLDER
    )

    final_dataset = sharegpt_openphish + sharegpt_tranco
    with open(OD_OUTPUT_FILE, 'w') as f:
        json.dump(final_dataset, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
