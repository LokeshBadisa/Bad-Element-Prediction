import base64
import re
import shutil
import threading
import concurrent.futures
from openai import OpenAI
from prompts import IMAGE_QUALITY_CHECK_SYSTEM_PROMPT
from tqdm import tqdm
from utils import *

SEMAPHORE_LIMIT = 30
FOLDER_WORKERS = 30
_sem = threading.Semaphore(SEMAPHORE_LIMIT)


BASE_DIR = '/data1/lokesh/shubho'

client = OpenAI(
    base_url="http://localhost:8995/v1",
    api_key="EMPTY"
)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def check_image_loaded(image_path):
    """Returns True if image is usable (page loaded), False if unusable (didn't load)."""
    try:
        response = client.chat.completions.create(
            model="gemma",
            messages=[
                {"role": "system", "content": IMAGE_QUALITY_CHECK_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
                        {"type": "text", "text": "Classify this webpage screenshot."}
                    ]
                }
            ],
            max_tokens=4096
        )
        response_text = response.choices[0].message.content
        match = re.search(r'"verdict"(.*?)"confidence"', response_text, re.DOTALL)
        if match:
            verdict_block = match.group(1).strip()
            value = re.search(r'"(.*?)"', verdict_block, re.DOTALL).group(1)
            return value == "usable"
        return True
    except Exception as e:
        print(f"Error checking image quality for {image_path}: {e}")
        return True


def _check_image_loaded_sem(image_path):
    with _sem:
        return check_image_loaded(image_path)


def all_images_same(key_list, folder_path, threshold=0.9):
    """Return True if all result images in key_list are pairwise similar (SSIM > threshold)."""
    if len(key_list) <= 1:
        return True
    for i in range(len(key_list)):
        for j in range(i + 1, len(key_list)):
            try:
                ssim_val = isSameScreenshot4(key_list[i], key_list[j], folder_path)[0]
                if ssim_val <= threshold:
                    return False
            except Exception:
                return False
    return True


shubho_usable = json.load(open('shubho_usable.json'))


def process_folder(folder):
    if folder.name not in shubho_usable:
        return    
    json_data = json.load(open(folder / 'data.json'))
    base_url = json_data.pop('base')

    filtered_items = [
        (number, box) for number, box in json_data.items()
        if 'error' not in box['url'].lower()
        and 'nothing changed and this is empty space' not in box['url'].lower()
        and 'url not found in url_dict' not in box['url'].lower()
        and Path(f'{folder}/screenshots/{number}.jpg').exists()
    ]
    anns = [{
        "x": box['x'],
        "y": box["y"],
        "width": box["width"],
        "height": box["height"]
    } for _, box in filtered_items]
    box_numbers = [number for number, _ in filtered_items]
    roots, TOTAL_NODES = build_bounding_box_tree(anns, box_numbers)
    for root in roots:
        compute_heights(root)

    LargeMatching, node_storage, isomorphs = match(roots)

    box_status = {}
    all_same_cache = {}
    renamed_parents = set()
    folder_path = f'/data1/lokesh/shubho/{folder.name}'

    # Pre-pass: find which parents actually have a child with SSIM <= 0.9
    needs_check = set()
    for k in isomorphs:
        node = find_node_given_roots(roots, k)
        key_list = sorted(node.get_keep_boxes().keys(), key=lambda x: int(x))
        key_list.remove(k)
        for key in key_list:
            try:
                if isSameScreenshot4(k, key, folder_path)[0] < 0.9:
                    needs_check.add(k)
                    break
            except Exception:
                pass

    # Run check_image_loaded only for parents that need it
    paths_to_check = {k: f'{folder_path}/screenshots/{k}.jpg' for k in needs_check}
    parent_loaded_cache = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=SEMAPHORE_LIMIT) as executor:
        future_to_k = {executor.submit(_check_image_loaded_sem, path): k
                       for k, path in paths_to_check.items()}
        for future in concurrent.futures.as_completed(future_to_k):
            k = future_to_k[future]
            try:
                parent_loaded_cache[paths_to_check[k]] = future.result()
            except Exception as e:
                print(f"Error checking image quality for {paths_to_check[k]}: {e}")
                parent_loaded_cache[paths_to_check[k]] = True

    for k, v in isomorphs.items():
        node = find_node_given_roots(roots, k)
        key_list = sorted(node.get_keep_boxes().keys(), key=lambda x: int(x))
        key_list.remove(k)
        ssim_dict = {}
        for key in key_list:
            try:
                ssim_dict[key] = isSameScreenshot4(k, key, folder_path)[0]
                if ssim_dict[key] > 0.9:
                    box_status[key] = {'status': 'remove', 'parent': k, 'ssim': ssim_dict[key]}
                else:
                    parent_img_path = f'{folder_path}/screenshots/{k}.jpg'
                    usability = parent_loaded_cache.get(parent_img_path, True)

                    if not usability:
                        all_same = all_images_same(key_list, folder_path)

                        if all_same:
                            #remove children
                            for temp_key in key_list:
                                box_status[temp_key] = {'status': 'remove', 'parent': k, 'ssim': ssim_dict[temp_key]}

                            #rename child to parent
                            shutil.copy(f'{folder_path}/screenshots/{key_list[0]}.jpg', f'{folder_path}/screenshots/{k}.jpg')
                        else:
                            # Children have different results: keep parent box
                            box_status[k] = {'status': 'remove', 'parent': k, 'ssim': None}

                    break
            except Exception as e:
                print(f"Error comparing {k} and {key} in folder {folder.name}: {e}")
                box_status[key] = {'status': 'keep', 'parent': k, 'ssim': 0.0}

    for getter in LargeMatching:
        if LargeMatching[getter] in box_status and box_status[LargeMatching[getter]]['status'] == 'remove':
            # json_data[getter]['status'] = 'remove'
            box_status[getter] = {'status': 'remove', 'parent': LargeMatching[getter], 'ssim': None}

    for key in box_status.keys():
        if box_status[key]['status'] == 'remove':
            json_data[key]['status'] = 'remove'

    if len(box_status.keys()) == 0:
        return

    json_data['base'] = base_url
    # with open(f'{folder}/data.json', 'w') as f:
    #     json.dump(json_data, f)
    Path(f'ssim_elimination/{folder.name}').mkdir(parents=True, exist_ok=True)
    for box, status in box_status.items():
        with open(f'ssim_elimination/{folder.name}/{box}.json', 'w') as f:
            json.dump(status, f)


folders = sorted(Path(BASE_DIR).iterdir(), key=lambda x: int(x.name))
with concurrent.futures.ThreadPoolExecutor(max_workers=FOLDER_WORKERS) as executor:
    futures = [executor.submit(process_folder, folder) for folder in folders]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="folders"):
        try:
            future.result()
        except Exception as e:
            print(f"Error processing folder: {e}")
