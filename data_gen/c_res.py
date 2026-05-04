from pathlib import Path
from tqdm import tqdm
import json
import re
import base64
from openai import OpenAI
from prompts import CONFLICT_RESOLUTION_SYSTEM_PROMPT
from preprocess_utils import SoM
from labelling_utils import deriveUrlFeatures

# Initialize OpenAI client
client = OpenAI(
    base_url="http://localhost:8995/v1",  # your local server
    api_key="EMPTY"
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


REASONING_FOLDER = 'gemma_vlm1_reasoning_wobf_final'
D1 = {}
for file in Path(REASONING_FOLDER).iterdir():
    json_data = json.load(file.open())
    filename = file.name.split('.')[0]
    D1[filename] = {}
    for key in json_data.keys():
        if key.split('_')[1] not in D1[filename]:
            D1[filename][key.split('_')[1]] = []     
        D1[filename][key.split('_')[1]].append(key)
    
D2 = {}
for filename in D1.keys():
    json_data = json.load(open(f'{REASONING_FOLDER}/{filename}.json'))
    D2[filename] = {}
    for k,v in D1[filename].items():
        if len(v) == 1:
            continue        
        for key in v:
            if key.split('_')[1] not in D2[filename]:
                D2[filename][key.split('_')[1]] = set()            
            D2[filename][key.split('_')[1]].add(re.findall(r'<answer>(.*?)</answer>', json_data[key])[-1])

total_conflicts = 0
for d in D2.values():
    for v in d.values():
        if len(v) > 1:
            total_conflicts += 1

pbar = tqdm(total=total_conflicts)            



for filename in D2.keys():
    if any(len(v) > 1 for v in D2[filename].values()):
        folder = f'/data1/lokesh/combineddata/{filename}'
        sommer = SoM(f'{folder}/base_screenshots', Path(f'{folder}/data.json'),\
                    f'{folder}/base_screenshots/metadata.json',\
                    process_each_box=True,process_eb_folder='./temp_each_box',\
                    crop_boxes=True,crop_location=f'./box_crops/{filename}')
        scroll_info = json.load(open(f'{folder}/base_screenshots/metadata.json'))['scroll_steps']
        answers_dict = {}
        box_info = json.load(open(f'{folder}/data.json'))
    else:
        continue

    for k,v in D2[filename].items():
        if len(v) > 1:
            coords, n_coords, img_sizes = [], [], []
            user_dict = []
            for box in D1[filename][k]:
                img_num = int(box.split('_')[0])          
                box_num = box.split('_')[1]      
                # try:                
                modified_y = box_info[box_num]["y"] - sum(scroll_info[:img_num])
                # except Exception as e:
                #     print(f"Error {e}")
                #     print(box_num, box_info.keys())

                c_x = box_info[box_num]['x']+box_info[box_num]['width']/2
                c_y = modified_y+box_info[box_num]['height']/2
                n_x = c_x/sommer.outputs[img_num].width
                n_y = c_y/sommer.outputs[img_num].height
                coords.append((c_x,c_y))
                n_coords.append((n_x,n_y))
                img_sizes.append((sommer.outputs[img_num].width, sommer.outputs[img_num].height))
                user_dict.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'temp_each_box/{img_num}/{box_num}.jpg')}"}})

            url1 = box_info['base']
            url2 = box_info[box_num]['url']
            boxes = sorted(D1[filename][k],key=lambda x: int(x.split('_')[0]))
            
            #values: D1[k] gives all values of same box
            prompt_text = f"Boxes are highlighted with {[sommer.color_list[box] for box in boxes]} colors in {range(1,len(boxes)+1)} images respectively. "+\
                    f"Center of the highlighted boxes are {coords} in raw pixel space. Image Sizes are {img_sizes}. "+\
                    f"Normalized Center of the highlighted box is {n_coords}. "+\
                    f"Image ids are {boxes}. "+\
                    f"URL of {range(1,len(boxes)+1)} images are {box_info['base']}. URL of second image is {box_info[box_num]['url']}. "+\
                    f"Derived features from URLs of both images are {deriveUrlFeatures(url1,url2)}."+\
                    f"did_anything_download = False"

            response = client.chat.completions.create(
                model="gemma",  # your local model
                messages=[
                    {"role": "system", "content": CONFLICT_RESOLUTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            *user_dict,
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'{folder}/screenshots/{box_num}.jpg')}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'./box_crops/{filename}/{box_num}.jpg')}"}},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ],
                max_tokens=5120
            )

            message = response.choices[0].message
            response_text = message.content
            if hasattr(message, "reasoning") and message.reasoning:
                reasoning = message.reasoning
                response_text += f"\n\n[Reasoning]: {reasoning}"
            answers_dict[k] = response_text
            pbar.update(1)

    Path('conflict_resolution_outputs').mkdir(exist_ok=True)
    with open(f'conflict_resolution_outputs/{filename}.json', 'w') as f:
        json.dump(answers_dict, f, indent=4)