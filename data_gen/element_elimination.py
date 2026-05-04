'''
This is model based box elimination.
Heuristic based box elimination is in utils.py
'''
from utils import *
from prompts import *
import json
import asyncio
from pathlib import Path
from PIL import Image
import base64
from openai import AsyncOpenAI
from preprocess_utils import *

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = AsyncOpenAI(
    base_url="http://localhost:8995/v1",
    api_key="EMPTY"
)

BASE_DIR = '/data1/lokesh/shubho'
SAVE_DIR = 'box_elimination_shubho'
CONCURRENCY = 20

semaphore = asyncio.Semaphore(CONCURRENCY)

def get_color(vis_img,x1,y1,x2):
    outmidpoint = ((x1+x2)/2, y1-2)
    inmidpoint = ((x1+x2)/2, y1+2)
    color1 = vis_img.img[int(inmidpoint[1]), int(inmidpoint[0])]/255.0
    color2 = vis_img.img[int(outmidpoint[1]), int(outmidpoint[0])]/255.0
    color = get_vibrant_separator(color1, color2)
    return describe_color(color)

async def process_folder(folder):
    async with semaphore:
        if Path(f'{SAVE_DIR}/{folder.name}/answers.json').exists():
            return
        json_data = json.load(open(folder / 'data.json'))
        base_url = json_data.pop('base')
        filtered_items = [
            (number, box) for number, box in json_data.items()
            if 'error' not in box['url'].lower()
            and 'nothing changed and this is empty space' not in box['url'].lower()
            and 'url not found in url_dict' not in box['url'].lower()
            and Path(f'{folder}/screenshots/{number}.jpg').exists()
            and Path(f'temp_shubho/{folder.name}/{number}.jpg').exists()
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
        total = len([v for k, v in isomorphs.items() if len(v) > 1 and find_node_given_roots(roots, k).height > 0])
        if total == 0:
            return
        if Path(f'{SAVE_DIR}/{folder.name}/answers.json').exists():
            return
        
        # sommer = SoM(f'{BASE_DIR}/{folder.name}/base_screenshots',\
        #             Path(f'{BASE_DIR}/{folder.name}/data.json'),\
        #             f'{BASE_DIR}/{folder.name}/base_screenshots/metadata.json',
        #             process_each_box=True,process_eb_folder=f'./temp_shubho') 
        # return
        pbar = tqdm(total=total)
        answers_dict = {}

        base_img = Image.open(f'{folder}/screenshot.jpg')
        vis_img = VisImage(np.asarray(base_img).clip(0, 255).astype(np.uint8), scale=1.0)

        async def process_isomorph(k, v):
            node = find_node_given_roots(roots, k)
            if len(v) <= 1 or node.height == 0:
                return

            user_dict_content = []
            user_dict_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'{folder}/final_boxes.jpg')}"}
            })
            

            key_list = sorted(node.get_keep_boxes().keys(), key=lambda x: int(x))
            coordinates = {key: {
                "x": json_data[key]['x'] + json_data[key]['width'] // 2,
                "y": json_data[key]['y'] + json_data[key]['height'] // 2,
                # "norm_x": (json_data[key]['x'] + json_data[key]['width'] // 2) / base_img.width,
                # "norm_y": (json_data[key]['y'] + json_data[key]['height'] // 2) / base_img.height,
                "top_left": (json_data[key]['x'], json_data[key]['y']),
                "bottom_right": (json_data[key]['x'] + json_data[key]['width'], json_data[key]['y'] + json_data[key]['height']),
                "tl_norm": (json_data[key]['x'] / base_img.width, json_data[key]['y'] / base_img.height),
                "br_norm": ((json_data[key]['x'] + json_data[key]['width']) / base_img.width,
                             (json_data[key]['y'] + json_data[key]['height']) / base_img.height),
                "color": get_color(vis_img, json_data[key]['x'], json_data[key]['y'], json_data[key]['x'] + json_data[key]['width'])
            } for key in key_list}
            
            for key in key_list:                
                # base_img.crop((
                #     json_data[key]['x'], json_data[key]['y'],
                #     json_data[key]['x'] + json_data[key]['width'],
                #     json_data[key]['y'] + json_data[key]['height']
                # )).save(f'{SAVE_DIR}/{folder.name}/{key}.jpg')
                user_dict_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'temp_shubho/{folder.name}/{key}.jpg')}"}
                })
                user_dict_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'{folder}/screenshots/{key}.jpg')}"}
                })

            prompt_text = (
                f"Boxes to be focused are {key_list}. Outermost Box is {k}."
                f"Image size is {base_img.width} and {base_img.height}."
                f"Top-left and bottom-right corners of box in pixel space are "
                f"{ {key: (coordinates[key]['top_left'], coordinates[key]['bottom_right']) for key in key_list} }"
                f"Top-left and bottom-right corners of box in normalized space are "
                f"{ {key: (coordinates[key]['tl_norm'], coordinates[key]['br_norm']) for key in key_list} }."
                f"Description of the color used to highlight boxes are { {key: coordinates[key]['color'] for key in key_list} }."
                f"URL of the first image and first image in each pair is same and it is {base_url}"
                f"URL list of the second image: {[json_data[key]['url'] for key in key_list]}"
                f"Identify which boxes can be removed."
            )
            user_dict_content.append({"type": "text", "text": prompt_text})

            response = await client.chat.completions.create(
                model="gemma",
                messages=[
                    {"role": "system", "content": BOX_ELIMINATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_dict_content}
                ],
                max_tokens=4096
            )

            message = response.choices[0].message
            response_text = message.content
            if hasattr(message, "reasoning") and message.reasoning:
                response_text += f"\n\n[Reasoning]: {message.reasoning}"

            answers_dict[k] = response_text
            pbar.update(1)

        # Process all isomorphs within a folder sequentially (inner loop unchanged)
        for k, v in isomorphs.items():
            await process_isomorph(k, v)

        if len(answers_dict.keys()) > 0:
            Path(f'{SAVE_DIR}/{folder.name}').mkdir(parents=True, exist_ok=True)
            with open(f'{SAVE_DIR}/{folder.name}/answers.json', 'w') as f:
                json.dump(answers_dict, f, indent=4)
        # shutil.rmtree(f'./temp/{folder.name}', ignore_errors=True)


async def main():
    folders = sorted(Path(BASE_DIR).iterdir(), key=lambda x: int(x.name))
    tasks = [process_folder(folder) for folder in folders]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())