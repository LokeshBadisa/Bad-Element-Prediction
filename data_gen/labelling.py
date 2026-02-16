from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
from qwen_agent.agents import Assistant # type: ignore
from qwen_agent.utils.output_beautify import multimodal_typewriter_print # type: ignore

llm_cfg = {
    # Use dashscope API
    # 'model': 'qwen3-vl-plus',
    # 'model_server': 'qwenvl_dashscope',
    # 'api_key': '' # **fill your api key here**

    # Use a model service compatible with the OpenAI API, such as vLLM or Ollama:
    'model_type': 'qwenvl_oai',
    'model': 'qwen',
    'model_server': 'http://localhost:9000/v1',  # base_url, also known as api_base
    'api_key': 'EMPTY',
    'generate_cfg': {
        "top_p": 0.8,
        "top_k": 20,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5
    }
}

analysis_prompt = """You are a security analysis model.

You are given two images:
1. The first image shows a webpage with a red box highlighting a specific button.
2. The second image shows the webpage after clicking that button.

Your task is to determine whether the highlighted button is malicious or not.

A button is MALICIOUS if clicking it results in any of the following:
- Redirect to scam, phishing, fake login, or deceptive page
- Automatic download of unknown or suspicious files
- Unexpected redirects to unrelated or spam websites
- Fake system warnings, fake updates, or fake antivirus pages
- Requests for sensitive information (passwords, OTPs, credit cards)
- Crypto scams, giveaway scams, or investment traps
- Popups or pages attempting social engineering

A button is NOT MALICIOUS if:
- It performs the expected legitimate action (navigation, submit, play, open content)
- It leads to normal site content or known legitimate services
- It triggers standard UI behavior without deception


Respond in following strict format with no extra text:
<answer>malicious or benign</answer>
"""

# tools = ['image_zoom_in_tool']

agent = Assistant(
    llm=llm_cfg,
    # function_list=tools,
    system_message=analysis_prompt,
    # [!Optional] We provide `analysis_prompt` to enable VL conduct deep analysis. Otherwise use system_message='' to simply enable the tools.
)


def get_img_num(y,height, num_imgs):    
    centers = [540*i + 540 for i in range(num_imgs)]
    box_center = y + height / 2
    min_dist = float('inf')
    img_num = -1
    for i, center in enumerate(centers):
        dist = abs(center - box_center)
        if dist < min_dist:
            min_dist = dist
            img_num = i
    return img_num

def highlight_box_on_image(img_path, x,y,width,height):    
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + width, y + height], outline="red", width=2)
    img.save("tmp_img.jpg")

#given a text, extract text between <answer> and </answer>
import re
def extract_answer(text):
    match = re.findall(r'<answer>(.*?)</answer>', text)
    # print(match)
    return match


for folder in sorted(Path('appropriate').iterdir()):
    # print(f'{folder}/{folder.name}/answers.json')
    if Path(f'{folder}/{folder.name}/answers.json').exists():
        continue
    json_data = json.load(open(f'{folder}/{folder.name}/data.json'))
    answers_dict = {}
    for key in tqdm(json_data):
        if "error: " in json_data[key]["url"]:
            continue
        base_img_num = get_img_num(json_data[key]["y"], json_data[key]["height"], len(list(Path(f"{folder}/{folder.name}/base_screenshots").glob("*.jpg"))))
        img_path = f'{folder}/{folder.name}/base_screenshots/{base_img_num:04d}.jpg'
        modified_y = json_data[key]["y"] - base_img_num*540

        highlight_box_on_image(img_path, json_data[key]["x"], modified_y, json_data[key]["width"], json_data[key]["height"])
        
        if not Path(f"{folder}/{folder.name}/screenshots/{key}.jpg").exists():
            continue
        messages = []
        messages += [
            {"role": "user", "content": [
                {"image": "tmp_img.jpg"},
                {"image": f"{folder}/{folder.name}/screenshots/{key}.jpg"},
                {"text": f"Analyze the highlighted button using the two images. Center of the red rectangle is {json_data[key]['x']+json_data[key]['width']/2} and {modified_y+json_data[key]['height']/2}."}
            ]}
        ]
        
        response_plain_text = ''
        for ret_messages in agent.run(messages):
            # `ret_messages` will contain all subsequent messages, consisting of interleaved assistant messages and tool responses
            response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
        # final_messages = list(agent.run(messages))[0]
        # print(final_messages)
        answer = extract_answer(response_plain_text)
        answers_dict[key] = answer
    with open(f'{folder}/{folder.name}/answers.json', 'w') as f:
        json.dump(answers_dict, f)    