from pathlib import Path
from tqdm import tqdm
import json
import shutil
from labelling_utils import *
from prompts import LABEL_GENERATION_SYSTEM_PROMPT
from preprocess_utils import SoM
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# client = OpenAI(
#     base_url="http://localhost:8995/v1",  # your local server
#     api_key="EMPTY"
# )



def isacceptable(folder,box):
    # D = [['13','5'],['13','6'],['13','11'],['13','23'],['13','29'],['13','40'],]#['5','7'],['7','6'],['10','4'],['10','70']
    # D = [['17','11'],['17','17'],['17','23'],['17','29'],]
    # D = [['10','4'],['13','6']]
    # D = [['17','5'],['17','6'],['17','35']]
    D = [['4','3'],['4','5']]
    for d in D:
        if folder==d[0] and box==d[1]:
            return True
    return False



def main():
    
    for folder in tqdm(sorted([folder for folder in Path('/data1/lokesh/combineddata').iterdir()], key=lambda x: int(x.name))[:20]):
        
        
        
        json_data = json.load(open(f'{folder}/data.json'))    
        scroll_info = json.load(open(f'{folder}/base_screenshots/metadata.json'))['scroll_steps']
        answers_dict = {}
        sommer = SoM(f'{folder}/base_screenshots', Path(f'{folder}/data.json'),\
                     f'{folder}/base_screenshots/metadata.json',\
                     process_each_box=True,process_eb_folder='./temp_each_box',\
                     crop_boxes=True,crop_location=f'./box_crops/{folder.name}')
        # curr_folder_number = folder.name

        #make a tqdm counter for the boxes in the image
        pbar = tqdm(total=sum([len(boxes) for boxes in sommer.boxes_in_image.values()]))
        for img_num, boxes_list in sommer.boxes_in_image.items():
            for box in boxes_list:
                if "error: " in json_data[box]["url"]:
                    continue    
                
                # if not isacceptable(folder.name, box):
                #     pbar.update(1)
                #     continue
                modified_y = json_data[box]["y"] - sum(scroll_info[:img_num])

                url1 = json_data['base']
                url2 = json_data[box]['url']
                
                
                prompt_text = f"Box is highlighted with {sommer.color_list[f'{img_num}_{box}']} color. "+\
                    f"Center of the highlighted box is {json_data[box]['x']+json_data[box]['width']/2} and {modified_y+json_data[box]['height']/2} in raw pixel space. Image Size is {sommer.outputs[img_num].width} and {sommer.outputs[img_num].height}. "+\
                    f"Normalized Center of the highlighted box is {(json_data[box]['x']+json_data[box]['width']/2)/sommer.outputs[img_num].width} and {(modified_y+json_data[box]['height']/2)/sommer.outputs[img_num].height}. "+\
                    f"URL of first image is {json_data['base']}. URL of second image is {json_data[box]['url']}. "+\
                    f"Derived features from URLs of both images are {deriveUrlFeatures(url1,url2)}."+\
                    f"did_anything_download = False"

                response = client.chat.completions.create(
                model="gemma",  # your local model
                messages=[
                    {"role": "system", "content": LABEL_GENERATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'temp_each_box/{img_num}/{box}.jpg')}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'{folder}/screenshots/{box}.jpg')}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'./box_crops/{folder.name}/{box}.jpg')}"}},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ],
                max_tokens=5120
            )
                

        #         message = response.choices[0].message
        #         response_text = message.content
        #         if hasattr(message, "reasoning") and message.reasoning:
        #             reasoning = message.reasoning
        #             response_text += f"\n\n[Reasoning]: {reasoning}"
        #         answers_dict[f"{img_num}_{box}"] = response_text
                pbar.update(1)
            
        # shutil.rmtree(f'./temp_each_box')
        # Path(f'gemma_vlm1_reasoning_wobf_testing').mkdir(parents=True, exist_ok=True)
        # with open(f'gemma_vlm1_reasoning_wobf_testing/{folder.name}.json', 'w') as f:
        #     json.dump(answers_dict, f, indent=4)    

               

if __name__ == "__main__":
    main()
