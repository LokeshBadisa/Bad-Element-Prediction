'''
This is model based box elimination. 
Heuristic based box elimination is in utils.py
'''
from utils import *
from prompts import *
import json
from pathlib import Path
from PIL import Image
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = OpenAI(
    base_url="http://localhost:8995/v1",  # your local server
    api_key="EMPTY"
)

for folder in sorted(Path('/data1/lokesh/combineddata').iterdir(),key=lambda x: int(x.name)):
    json_data = json.load(open(folder/'data.json'))
    json_data.pop('base')
    anns = [{
                    "x": box['x'],
                    "y": box["y"],
                    "width": box["width"],
                    "height": box["height"]             
                } for box in json_data.values()]
    box_numbers = list(json_data.keys())
    roots, TOTAL_NODES = build_bounding_box_tree(anns, box_numbers)     
    for root in roots:
        compute_heights(root)
    
    LargeMatching, node_storage, isomorphs = match(roots)
    pbar = tqdm(total=len([v for k,v in isomorphs.items() if len(v) > 1 and find_node_given_roots(roots, k).height > 0]))
    answers_dict = {}

    base_img = Image.open(f'{folder}/screenshot.jpg')
    for k, v in isomorphs.items():
        node = find_node_given_roots(roots, k)
        if len(v) <= 1 or node.height == 0:
            continue
        
        
        user_dict_content = []
        
        user_dict_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'{folder}/final_boxes.jpg')}"}},)
        key_list = sorted(node.get_keep_boxes().keys(), key=lambda x: int(x))
        coordinates = {key: {
                        "x": json_data[key]['x']+json_data[key]['width']//2,
                        "y": json_data[key]['y']+json_data[key]['height']//2,
                        "norm_x": (json_data[key]['x']+json_data[key]['width']//2)/base_img.width,
                        "norm_y": (json_data[key]['y']+json_data[key]['height']//2)/base_img.height
                        } for key in key_list}
        for key in key_list:
            Path(f'box_elimination/{folder.name}').mkdir(parents=True, exist_ok=True)
            base_img.crop((json_data[key]['x'], json_data[key]['y'], json_data[key]['x']+json_data[key]['width'], json_data[key]['y']+json_data[key]['height'])).save(f'box_elimination/{folder.name}/{key}.jpg')
            user_dict_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'box_elimination/{folder.name}/{key}.jpg')}"}},)
            user_dict_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'{folder}/screenshots/{key}.jpg')}"}},)
        prompt_text = f"Boxes to be focused are {key_list}."+\
                    f"Outermost Box is {k}."+\
                    f"Coordinates of centers of boxes: { {key: (coordinates[key]['x'], coordinates[key]['y']) for key in key_list} }"+\
                    f"Image size is {base_img.width} and {base_img.height}."+\
                    f"Normalized coordinates of centers of boxes are { {key: (coordinates[key]['norm_x'],coordinates[key]['norm_y']) for key in key_list} }."
        user_dict_content.append({"type": "text", "text": prompt_text})
        response = client.chat.completions.create(
                model="gemma",  # your local model
                messages=[
                    {"role": "system", "content": BOX_ELIMINATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_dict_content
                    }
                ],
                max_tokens=4096
            )

        message = response.choices[0].message
        response_text = message.content
        if hasattr(message, "reasoning") and message.reasoning:
            reasoning = message.reasoning
            response_text += f"\n\n[Reasoning]: {reasoning}"
        answers_dict[k] = response_text
        pbar.update(1)
    
    Path(f'box_elimination/{folder.name}').mkdir(parents=True, exist_ok=True)
    with open(f'box_elimination/{folder.name}/answers.json', 'w') as f:
        json.dump(answers_dict, f, indent=4)