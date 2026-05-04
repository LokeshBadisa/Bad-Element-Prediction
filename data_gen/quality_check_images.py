from pathlib import Path
from tqdm import tqdm
import json
import shutil
from preprocess_utils import extract_answer
from prompts import IMAGE_QUALITY_CHECK_SYSTEM_PROMPT
import re
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client = OpenAI(
    base_url="http://localhost:8995/v1",  # your local server
    api_key="EMPTY"
)


with open('shubho_tranco.json', 'r') as f:
    folder_list = json.load(f)
BASE_DIR = '/data1/lokesh/tranco_data/bep/data_gen/data'
SAVE_DIR = 'NEW_DIR'


for folder_name in tqdm(folder_list[:10]):    
    if Path(f'{SAVE_DIR}/{folder_name}/image.json').exists():
        continue
    prompt_text = "Classify this webpage screenshot."

    response = client.chat.completions.create(
        model="gemma",  # your local model
        messages=[
            {"role": "system", "content": IMAGE_QUALITY_CHECK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f'{BASE_DIR}/{folder_name}/screenshot.jpg')}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        max_tokens=4096
    )
    
    response_plain_text = response.choices[0].message.content
    answer = re.search(r'"verdict"(.*?)"confidence"', response_plain_text, re.DOTALL)
    verdict_block = answer.group(1).strip()

    # Extract text between quotes
    value = re.search(r'"(.*?)"', verdict_block, re.DOTALL).group(1)    
    # answer = extract_answer(response.choices[0].message.content)[-1]
     
    Path(f'{SAVE_DIR}/{folder_name}').mkdir(parents=True, exist_ok=True)
    with open(f'{SAVE_DIR}/{folder_name}/image.json', 'w') as f:
        json.dump(value, f)
    