from tqdm import tqdm
import json
from pathlib import Path
from prompts import *
from preprocess_utils import SoM, numpy_to_base64 
from qwen_agent.agents import Assistant 
from qwen_agent.utils.output_beautify import multimodal_typewriter_print 


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
    system_message=BOX_QUALITY_CHECK_SYSTEM_PROMPT,
    # [!Optional] We provide `analysis_prompt` to enable VL conduct deep analysis. Otherwise use system_message='' to simply enable the tools.
)

list_to_process = []

for folder in sorted(Path('/data1/lokesh/combineddata').iterdir(),key=lambda x: int(x.name)):
    image_check = json.load(open(f'{folder}/quality_check/image.json'))
    if image_check[0]=='complete':
        list_to_process.append(folder)

list_to_process = sorted(list_to_process, key=lambda x: int(x.name))


for folder in tqdm(list_to_process):        

    sommer = SoM(folder/f'base_screenshots',folder/f'data.json',folder/f'base_screenshots/metadata.json')    

    for img_num in sommer.boxes_in_image.keys():
        # if Path(f'{folder}/quality_check/boxes/{img_num}.json').exists():
        #     continue
        img_file = sommer.outputs[img_num].get_image()        
        img_file = numpy_to_base64(img_file)
        # print(f'Boxes in {img_num}: {sommer.boxes_in_image[img_num]}')
        messages = BOX_QUALITY_CHECK_FEW_SHOT_EXAMPLES.copy() # start with few shot examples
        messages += [
            {"role": "user", "content": [            
                {"image": f"data:image/jpeg;base64,{img_file}"},
                {"text": f"The image contains bounding box drawn over possible UI element. Boxes in image are {sommer.boxes_in_image[img_num]}. Follow the required output format strictly."}
            ]}
        ]

        response_plain_text = ''
        for ret_messages in agent.run(messages):
            # `ret_messages` will contain all subsequent messages, consisting of interleaved assistant messages and tool responses
            response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)

        Path(f'{folder}/quality_check/boxes').mkdir(parents=True, exist_ok=True)
        with open(f'{folder}/quality_check/boxes/{img_num}.json', 'w') as f:
            #extract text between {}
            extracted_text = response_plain_text[response_plain_text.find('{'):response_plain_text.rfind('}')+1]            
            json.dump(json.loads(extracted_text), f)                
            # json.dump(response_plain_text, f)

    sommer.save(folder/f'quality_check/boxes') 

    # break