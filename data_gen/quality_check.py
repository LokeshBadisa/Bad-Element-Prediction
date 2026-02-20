from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
import re
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

analysis_prompt = """You are a webpage quality inspector.

Given a screenshot of a webpage, decide whether the page is a properly rendered, usable content page or not.

A page is NOT usable if it shows:
- "Page not found", "404", "403", "500"
- CAPTCHA, bot verification, Cloudflare, "checking your browser"
- Blank/white page or loading spinner only
- Error messages or access denied

A page IS usable if it shows:
- Actual readable content (text, images, article, product, dashboard, etc.)
- Cookie banners and privacy consent notices are also okay
- Login walls, paywalls, consent-only screens are also okay

Respond in following strict format with no extra text:
<answer>appropriate or in-appropriate</answer>
"""

# tools = ['image_zoom_in_tool']

agent = Assistant(
    llm=llm_cfg,
    # function_list=tools,
    system_message=analysis_prompt,
    # [!Optional] We provide `analysis_prompt` to enable VL conduct deep analysis. Otherwise use system_message='' to simply enable the tools.
)


def extract_answer(text):
    match = re.findall(r'<answer>(.*?)</answer>', text)
    # print(match)
    return match


for folder in tqdm(sorted(Path('data2').iterdir())):    
    # if Path(f'{folder}/status.json').exists():
    #     continue
    # print(f'{folder}/status.json')
    json_data = json.load(open(f'{folder}/data.json'))
    answers_dict = {}
    messages = []
    messages += [
        {"role": "user", "content": [            
            {"image": f"{folder}/screenshot.jpg"},
            {"text": "Classify this webpage screenshot."}
        ]}
    ]
    response_plain_text = ''
    for ret_messages in agent.run(messages):
        # `ret_messages` will contain all subsequent messages, consisting of interleaved assistant messages and tool responses
        response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)

    answer = extract_answer(response_plain_text)
    with open(f'{folder}/status.json', 'w') as f:
        json.dump(answer, f)    