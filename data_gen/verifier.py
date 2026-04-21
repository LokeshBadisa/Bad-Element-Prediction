from pathlib import Path
from tqdm import tqdm
import json
import shutil
import base64
from openai import OpenAI

from labelling_utils import *
from prompts import CRITIC_SYSTEM_PROMPT
from preprocess_utils import SoM, get_img_num

# Initialize OpenAI client
client = OpenAI(
    base_url="http://localhost:9013/v1",  # your local server
    api_key="EMPTY"
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


for file in tqdm(sorted(Path('previous_reasoning').iterdir(), key=lambda x: int(x.stem))[:20]):
    if Path(f'verifier_reasoning/{file.stem}.json').exists():
        continue
    # if file.stem != '13':
    #     continue

    json_data = json.load(open(f'/data1/lokesh/combineddata/{file.stem}/data.json'))    
    scroll_info = json.load(open(f'/data1/lokesh/combineddata/{file.stem}/base_screenshots/metadata.json'))['scroll_steps']
    previous_reasoning = json.load(open(file))

    answers_dict = {}

    sommer = SoM(
        f'/data1/lokesh/combineddata/{file.stem}/base_screenshots',
        Path(f'/data1/lokesh/combineddata/{file.stem}/data.json'),
        f'/data1/lokesh/combineddata/{file.stem}/base_screenshots/metadata.json',
        process_each_box=True,
        process_eb_folder='./temp_each_box'
    )

    pbar = tqdm(total=sum([len(boxes) for boxes in sommer.boxes_in_image.values()]))

    for img_num, boxes_list in sommer.boxes_in_image.items():
        for box in boxes_list:

            if "error: " in json_data[box]["url"]:
                continue    

            if f"{img_num}_{box}" not in previous_reasoning:
                continue

            if len(previous_reasoning[f"{img_num}_{box}"]) <= 60:
                continue

            # if box !='37':
            #     continue

            modified_y = json_data[box]["y"] - sum(scroll_info[:img_num])

            url1 = json_data['base']
            url2 = json_data[box]['url']

            img1_path = f"temp_each_box/{img_num}/{box}.jpg"
            img2_path = f"/data1/lokesh/combineddata/{file.stem}/screenshots/{box}.jpg"

            if not Path(img2_path).exists():
                continue

            # Encode images
            img1_b64 = encode_image(img1_path)
            img2_b64 = encode_image(img2_path)

            prompt_text = (
                f"Box is highlighted with {sommer.color_list[f'{img_num}_{box}']} color."
                f"Center of the highlighted box is {json_data[box]['x']+json_data[box]['width']/2} and {modified_y+json_data[box]['height']/2} in raw pixel space. Image Size is {sommer.outputs[img_num].width} and {sommer.outputs[img_num].height}."
                f"Normalized Center of the highlighted box is {(json_data[box]['x']+json_data[box]['width']/2)/sommer.outputs[img_num].width} and {(modified_y+json_data[box]['height']/2)/sommer.outputs[img_num].height}."
                f"URL of first image is {url1}. "
                f"URL of second image is {url2}. "
                f"Derived features from URLs of both images are {deriveUrlFeatures(url1, url2)}. "
                f"Previous model reasoning for this box: {previous_reasoning[f'{img_num}_{box}']}"
            )

            response = client.chat.completions.create(
                model="gemma",  # your local model
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ],
                max_tokens=2048
            )

            message = response.choices[0].message
            response_text = message.content
            if hasattr(message, "reasoning") and message.reasoning:
                reasoning = message.reasoning
                response_text += f"\n\n[Reasoning]: {reasoning}"
            answers_dict[f"{img_num}_{box}"] = response_text

            pbar.update(1)

    shutil.rmtree('./temp_each_box')

    Path('./verifier_reasoning').mkdir(parents=True, exist_ok=True)

    with open(f'verifier_reasoning/{file.stem}.json', 'w') as f:
        json.dump(answers_dict, f, indent=4)