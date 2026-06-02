from pathlib import Path
from tqdm import tqdm
import json
from labelling_utils import *
from prompts import LABEL_GENERATION_SYSTEM_PROMPT
from preprocess_utils import SoM
import base64
import pandas as pd

df = []

BASE_DIR = '/data1/lokesh/v2_zip_openphish_usable'
folder_list = [folder for folder in json.load(open('retain_dict_may23.json')).keys() if not Path(f'GLR_json/{folder}.json').exists()]
folder_list = list(set(folder_list))
    
for folder in tqdm([Path(BASE_DIR + '/' + folder) for folder in folder_list if not Path(f'GLR_json/{folder}.json').exists()]):
    # if json.load(open(f'NEW_DIR_openphish_v2/{folder.name}/image.json')) !='usable':
    #     continue
    if not Path(f'{folder}/data.json').exists():
        continue
    json_data = json.load(open(f'{folder}/data.json'))    
    scroll_info = json.load(open(f'{folder}/base_screenshots/metadata.json'))['scroll_steps']
    
    sommer = SoM(f'{folder}/base_screenshots', Path(f'{folder}/data.json'),\
                    f'{folder}/base_screenshots/metadata.json',\
                    process_each_box=True,process_eb_folder=f'./temp_each_box/{folder.name}',\
                    crop_boxes=True,crop_location=f'./box_crops/{folder.name}')
    

    for img_num, boxes_list in sommer.boxes_in_image.items():
        for box in boxes_list:
            
            modified_y = json_data[box]["y"] - sum(scroll_info[:img_num])

            url1 = json_data['base']
            url2 = json_data[box]['url']
            
            adder = {
                        "id": f'{folder.name}_{img_num}_{box}',
                        "color":sommer.color_list[f'{img_num}_{box}'],
                        "center_x": json_data[box]['x']+json_data[box]['width']/2,
                        "center_y": modified_y+json_data[box]['height']/2,
                        "image_width": sommer.outputs[img_num].width,
                        "image_height": sommer.outputs[img_num].height,
                        "normalized_center_x": (json_data[box]['x']+json_data[box]['width']/2)/sommer.outputs[img_num].width,
                        "normalized_center_y": (modified_y+json_data[box]['height']/2)/sommer.outputs[img_num].height,
                        "url1": url1,
                        "url2": url2,
                        "derived_url_features": deriveUrlFeatures(url1, url2),
                        "did_anything_download": "True" if "isMalicious" in json_data[box] else "False",
                        "img_url1":f'temp_each_box/{folder.name}/{img_num}/{box}.jpg',
                        "img_url2":f'{folder}/screenshots/{box}.jpg',
                        "img_url3":f'./box_crops/{folder.name}/{box}.jpg'
                    }
            
            df.append(adder)


df = pd.DataFrame(df)
df.to_csv('reader_temp.csv', index=False)