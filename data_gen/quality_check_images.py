from pathlib import Path
from tqdm import tqdm
import json
import shutil
from preprocess_utils import extract_answer
from prompts import IMAGE_QUALITY_CHECK_SYSTEM_PROMPT
from qwen_agent.agents import Assistant # type: ignore
from qwen_agent.utils.output_beautify import multimodal_typewriter_print # type: ignore
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# llm_cfg = {
#     # Use dashscope API
#     # 'model': 'qwen3-vl-plus',
#     # 'model_server': 'qwenvl_dashscope',
#     # 'api_key': '' # **fill your api key here**

#     # Use a model service compatible with the OpenAI API, such as vLLM or Ollama:
#     'model_type': 'qwenvl_oai',
#     'model': 'qwen',
#     'model_server': 'http://localhost:9013/v1',  # base_url, also known as api_base
#     'api_key': 'EMPTY',
#     'generate_cfg': {
#         "top_p": 0.8,
#         "top_k": 20,
#         "temperature": 0.7,
#         "repetition_penalty": 1.0,
#         "presence_penalty": 1.5
#     }
# }
client = OpenAI(
    base_url="http://localhost:9013/v1",  # your local server
    api_key="EMPTY"
)


# tools = ['image_zoom_in_tool']

# agent = Assistant(
#     llm=llm_cfg,
#     # function_list=tools,
#     system_message=IMAGE_QUALITY_CHECK_SYSTEM_PROMPT,
#     # [!Optional] We provide `analysis_prompt` to enable VL conduct deep analysis. Otherwise use system_message='' to simply enable the tools.
# )
with open('shubho_data.json', 'r') as f:
    folder_list = json.load(f)
BASE_DIR = '/data1/lokesh/shubho'
SAVE_DIR = 'NEW_DIR'
# for folder in tqdm(sorted(Path('./data').iterdir(),key=lambda x: int(x.name))):    
    # if Path(f'{folder}/quality_check/image.json').exists():
    #     continue
for folder_name in tqdm(folder_list):    
    # if (folder / 'data.json').exists():
    #     jsondata = json.load((folder / 'data.json').open())
    #     if '-1' in jsondata or len(list(jsondata.keys())) == 1 or json.loads(open(f'{str(folder)}/status.json').read())[0] != 'appropriate':
    #         if len(list(jsondata.keys())) == 1:
    #             shutil.rmtree(folder)
    #         continue
    # else:
    #     continue

    # json_data = json.load(open(f'{folder}/data.json'))
    # answers_dict = {}
    # for key in tqdm(json_data.keys()):
    #     if key == 'base' or "error: " in json_data[key]["url"] or 'quality_check' in json_data[key]:
    #         continue
    # messages = []
    # messages += [
    #     {"role": "user", "content": [            
    #         # {"image": f"{folder}/screenshot.jpg"},
    #         {"image": f"{BASE_DIR}/{folder_name}/screenshot.jpg"},
    #         {"text": "Classify this webpage screenshot."}
    #     ]}
    # ]
    # response_plain_text = ''
    # try:
    #     for ret_messages in agent.run(messages):
    #         # `ret_messages` will contain all subsequent messages, consisting of interleaved assistant messages and tool responses
    #         response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
    # except Exception as e:
    #     continue
    # answer = extract_answer(response_plain_text)[-1]
        # json_data[key]['quality_check'] = [answer]
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
    answer = extract_answer(response_plain_text)[-1]
     
    Path(f'{SAVE_DIR}/{folder_name}').mkdir(parents=True, exist_ok=True)
    with open(f'{SAVE_DIR}/{folder_name}/image.json', 'w') as f:
        json.dump(extract_answer(response), f)
    