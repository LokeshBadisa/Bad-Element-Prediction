from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
from prompts import *
from preprocess_utils import *
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


# tools = ['image_zoom_in_tool']

agent = Assistant(
    llm=llm_cfg,
    # function_list=tools,
    system_message=LABEL_GENERATION_SYSTEM_PROMPT,
    # [!Optional] We provide `analysis_prompt` to enable VL conduct deep analysis. Otherwise use system_message='' to simply enable the tools.
)




for folder in tqdm(sorted([folder for folder in Path('data2').iterdir() if json.load(open(f'{folder}/status.json'))[0] == 'appropriate'])):
    # print(f'{folder}/{folder.name}/answers.json')
    # if Path(f'{folder}/answers.json').exists():
    #     continue
    json_data = json.load(open(f'{folder}/data.json'))    
    scroll_info = json.load(open(f'{folder}/base_screenshots/metadata.json'))['scroll_steps']
    answers_dict = {}
    for key in tqdm([key for key in json_data.keys() if key != 'base']):
        if "error: " in json_data[key]["url"]:
            continue
        base_img_num = get_img_num(json_data[key]["y"], json_data[key]["height"], scroll_info)
        img_path = f'{folder}/base_screenshots/{base_img_num:04d}.jpg'
        modified_y = json_data[key]["y"] - sum(scroll_info[:base_img_num])

        highlight_box_on_image(img_path, json_data[key]["x"], modified_y, json_data[key]["width"], json_data[key]["height"])
        
        if not Path(f"{folder}/screenshots/{key}.jpg").exists():
            continue
        messages = []
        messages += [
            {"role": "user", "content": [
                {"image": "tmp_img.jpg"},
                {"image": f"{folder}/screenshots/{key}.jpg"},
                {"text": f"Analyze the highlighted button using the two images. Center of the red rectangle is {json_data[key]['x']+json_data[key]['width']/2} and {modified_y+json_data[key]['height']/2}. URL of first image is {json_data['base']}. URL of second image is {json_data[key]['url']}"}
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
    with open(f'{folder}/answers.json', 'w') as f:
        json.dump(answers_dict, f)    