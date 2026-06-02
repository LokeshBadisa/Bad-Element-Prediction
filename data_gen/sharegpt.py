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
SHAREGPT_OUTPUT_FILE = 'train_sharegpt_data_may20.json'
remove_list = []


# folder_list = open('train_set_may20.txt').read().splitlines()

def create_data(folder_list, TYPE, BASE_DIR, REASONING_FOLDER=None, CONFLICT_RES_FOLDER=None, BOX_CROP_FOLDER=None, FINAL_FOLDER=None):
    sharegpt = []
    remove_list = []
    for folder in tqdm([Path(f'{BASE_DIR}/{folder}') for folder in folder_list]):    
        
        jsondata = json.load(open(folder/f'data.json'))
        
        if TYPE == 'malicious':
            try:
                labelsdata = json.load(open(f'{REASONING_FOLDER}/{folder.name}.json'))
            except:
                remove_list.append(folder.name)
                continue
        else:
            labelsdata = defaultdict(lambda: ['benign'])
        scrolldata = json.load(open(folder/f'base_screenshots/metadata.json'))['scroll_steps']

        temp_dict_base = {"messages": [],"images": []}
        temp_dict_base["messages"].append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        )
        Path(f'{FINAL_FOLDER}/{folder.name}').mkdir(parents=True, exist_ok=True)
        sommer = SoM(folder/f'base_screenshots',folder/f'data.json',\
            folder/f'base_screenshots/metadata.json',process_all_boxes=True,\
                crop_boxes=True,crop_location=f'{BOX_CROP_FOLDER}/{folder.name}')#,True if folder.name in ['1904'] else False)    
        sommer.save(f'{FINAL_FOLDER}/{folder.name}')

        label_dict = {}
        for i in range(len(scrolldata)+1):
            boxes_in_image = sommer.boxes_in_image.get(i, [])
            

            for key in sorted(boxes_in_image):
                try:
                    if Path(f'{CONFLICT_RES_FOLDER}/{folder.name}_{key}.json').exists():
                        extracted = extract_labels(open(f'{CONFLICT_RES_FOLDER}/{folder.name}_{key}.json').read())                    
                    else:
                        extracted = extract_labels(labelsdata[f'{i}_{key}'])
                    
                    if 'obfuscated' in extracted:
                        sommer.boxes_in_image[i].remove(key)
                    else:    
                        label_dict[key] = extracted
                except Exception as e:
                    sommer.boxes_in_image[i].remove(key)

        for i in range(len(scrolldata)+1):
            
            boxes_in_image = sommer.boxes_in_image.get(i, [])
            if len(boxes_in_image) == 0:    
                continue
            
            temp_dict = copy.deepcopy(temp_dict_base)
                    
            color_dict = {box:sommer.color_list[f'{i}_{box}'] for box in boxes_in_image}
            center_ys = {box: jsondata[box]["y"] - sum(scrolldata[:i])+jsondata[box]['height']/2 for box in boxes_in_image}
            center_xs = {box: jsondata[box]["x"] + jsondata[box]['width']/2 for box in boxes_in_image}
            paired_centers = {box: (center_xs[box], center_ys[box]) for box in boxes_in_image}
            image_width = sommer.outputs[i].width
            image_height = sommer.outputs[i].height
            norm_xs = {box: (center_xs[box] / image_width) for box in boxes_in_image}
            norm_ys = {box: (center_ys[box] / image_height) for box in boxes_in_image}
            paired_norm_centers = {box: (norm_xs[box], norm_ys[box]) for box in boxes_in_image}
            extracted_feats = extract_all_features(jsondata['base'])
            #make a string such that <image> is repeated for len(boxes_in_image) times
            image_string = "<image> " * (len(boxes_in_image)+1)
            temp_dict["messages"].append(
                {
                    "role": "human",
                    "content": f"{image_string} Classify the elements in given image. URL of this page is {jsondata['base']}"+\
                        f"Boxes in the image are: {', '.join(boxes_in_image)}. "+\
                        f"Each box is highlighted with these colors: {color_dict}. "+\
                        f"Centers of the boxes are: {paired_centers}. "+\
                        f"Image size: {(image_width, image_height)}. "+\
                        f"Normalized centers of the boxes are: {paired_norm_centers}. "+\
                        f"Extracted features of the URL are: {extracted_feats}. "
                }
            )
            temp_dict["messages"].append(
                {
                    "role": "gpt",
                    "content": '```json{'+', '.join(f'{key}:{label_dict[key]}' for key in sorted(boxes_in_image))+'}```'
                }
            )
            temp_dict["images"].append(str(Path(f'{FINAL_FOLDER}/{folder.name}/{i}.jpg').absolute()))
            for box in boxes_in_image:
                temp_dict["images"].append(str(Path(f'{BOX_CROP_FOLDER}/{folder.name}/{box}.jpg').absolute()))
            sharegpt.append(temp_dict)
    return sharegpt


openphish_folder_list = open('train_set_may20_shubho.txt').read().splitlines()
tranco_folder_list = open('train_set_may20_tranco.txt').read().splitlines()
sharegpt_tranco = create_data(tranco_folder_list, 'benign', TRANCO_BASE_DIR, BOX_CROP_FOLDER=TRANCO_BOX_CROP_FOLDER, FINAL_FOLDER=TRANCO_FINAL_FOLDER)
sharegpt_openphish = create_data(openphish_folder_list, 'malicious', OPENPHISH_BASE_DIR, OPENPHISH_REASONING_FOLDER, OPENPHISH_CONFLICT_RES_FOLDER, OPENPHISH_BOX_CROP_FOLDER, OPENPHISH_FINAL_FOLDER)

final_sharegpt = sharegpt_openphish + sharegpt_tranco
with open(SHAREGPT_OUTPUT_FILE, "w") as f:
    json.dump(final_sharegpt, f, indent=4)

# with open('removed_folders.txt', 'w') as f:
#     f.write('\n'.join(remove_list))  