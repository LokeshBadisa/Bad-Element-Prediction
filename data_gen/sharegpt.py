import json
from tqdm import tqdm
from pathlib import Path
from preprocess_utils import SoM
from prompts import SYSTEM_PROMPT
import copy
from collections import defaultdict

TYPE = 'benign'
GIVEN_FOLDER_NAME = 'tranco_complete_data_sharegpt'

sharegpt = []
for folder in tqdm(sorted(Path('./data').iterdir())):
        
        # print(f"Processing folder: {folder}")
    # if Path(folder/'final_boxes.jpg').exists():        
        jsondata = json.load(open(folder/f'data.json'))
        
        if TYPE == 'malicious':
            labelsdata = json.load(open(folder/f'answers.json'))
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
        Path(f'{GIVEN_FOLDER_NAME}/{folder.name}').mkdir(parents=True, exist_ok=True)
        sommer = SoM(folder/f'base_screenshots',folder/f'data.json',folder/f'base_screenshots/metadata.json',process_all_boxes=True)#,True if folder.name in ['1904'] else False)    
        sommer.save(f'{GIVEN_FOLDER_NAME}/{folder.name}')
        # print(f"Saved processed images for folder: {GIVEN_FOLDER_NAME}/{folder.name}')")
        for i in range(len(scrolldata)+1):
            # boxes_in_image = [int(key) for key in labelsdata.keys() if sommer.inimagedict[(key, i)]]
            boxes_in_image = sommer.boxes_in_image.get(i, [])
            if len(boxes_in_image) == 0:    
                continue
            # for key in sorted(boxes_in_image):                 
            temp_dict = copy.deepcopy(temp_dict_base)
            # temp_dict["messages"].append(
            #     {
            #         "role": "user",
            #         "content": "<image>"
            #     }
            # )
            temp_dict["messages"].append(
                {
                    "role": "human",
                    "content": f"<image> Classify the elements in given image. URL of this page is {jsondata['base']}"
                }
            )
            temp_dict["messages"].append(
                {
                    "role": "gpt",
                    "content": '{'+', '.join(f'{key}:{labelsdata[str(key)][-1]}' for key in sorted(boxes_in_image))+'}'
                }
            )
            temp_dict["images"].append(f'{GIVEN_FOLDER_NAME}/{folder.name}/{i}.jpg')
            sharegpt.append(temp_dict)

with open(f"{GIVEN_FOLDER_NAME}/sharegpt_data.json", "w") as f:
    json.dump(sharegpt, f, indent=4)    