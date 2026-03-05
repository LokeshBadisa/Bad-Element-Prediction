import json
from tqdm import tqdm
from pathlib import Path
from utils import SoM
from prompts import SYSTEM_PROMPT
import copy


sharegpt = []
for folder in tqdm(sorted(Path('../../combineddata').iterdir())):
        if folder.name not in ['64']:
            continue
        print(f"Processing folder: {folder}")
    # if Path(folder/'final_boxes.jpg').exists():        
        jsondata = json.load(open(folder/f'data.json'))
        if 'base' not in jsondata:
            print(f"Skipping {folder} as 'base' key is missing in data.json")
            continue
        labelsdata = json.load(open(folder/f'answers.json'))
        scrolldata = json.load(open(folder/f'base_screenshots/metadata.json'))['scroll_steps']

        temp_dict_base = {"messages": [],"images": []}
        temp_dict_base["messages"].append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        )
        Path(f'traindata_wurl2/{folder.name}').mkdir(parents=True, exist_ok=True)
        sommer = SoM(folder/f'base_screenshots',folder/f'data.json',folder/f'base_screenshots/metadata.json')#,True if folder.name in ['1904'] else False)    
        sommer.save(f'traindata_wurl2/{folder.name}')
        print(f"Saved processed images for folder: traindata_wurl2/{folder.name}')")
        for i in range(len(scrolldata)+1):
            boxes_in_image = [int(key) for key in labelsdata.keys() if sommer.inimagedict[(key, i)]]
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
                    "content": f"<image> Classify the given image. URL of this page is {jsondata['base']}"
                }
            )
            temp_dict["messages"].append(
                {
                    "role": "gpt",
                    "content": '{'+', '.join(f'{key}:{labelsdata[str(key)][-1]}' for key in sorted(boxes_in_image))+'}'
                }
            )
            temp_dict["images"].append(f'/data1/lokesh/bep/data_gen/traindata_wurl/{folder.name}/{i}.jpg')
            sharegpt.append(temp_dict)

with open("traindata_wurl/sharegpt_data.json", "w") as f:
    json.dump(sharegpt, f, indent=4)    