from pathlib import Path
from tqdm import tqdm
import json
import shutil
from labelling_utils import *
from prompts import LABEL_GENERATION_SYSTEM_PROMPT
from preprocess_utils import SoM
# from qwen_agent.agents import Assistant # type: ignore
# from qwen_agent.utils.output_beautify import multimodal_typewriter_print # type: ignore
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
#     'model': 'gemma',
#     'model_server': 'http://localhost:9013/v1',  # base_url, also known as api_base
#     'api_key': 'EMPTY',
#     # 'generate_cfg': {
#     #     "top_p": 0.8,
#     #     "top_k": 20,
#     #     "temperature": 0.7,
#     #     "repetition_penalty": 1.0,
#     #     "presence_penalty": 1.5
#     # }
#     # 'generate_cfg': {
#     #     "top_p": 0.95,
#     #     "top_k": 20,
#     #     "temperature": 1.0,
#     #     "repetition_penalty": 1.0,
#     #     "presence_penalty": 1.5,        
#     # }
# }

client = OpenAI(
    base_url="http://localhost:8995/v1",  # your local server
    api_key="EMPTY"
)

#Qwen3.5: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0


# tools = ['image_zoom_in_tool']

# agent = Assistant(
#     llm=llm_cfg,
#     # function_list=tools,
#     system_message=LABEL_GENERATION_SYSTEM_PROMPT,
#     # [!Optional] We provide `analysis_prompt` to enable VL conduct deep analysis. Otherwise use system_message='' to simply enable the tools.
# )

def isacceptable(folder,box):
    D = [['13','5'],['13','6'],['13','11'],['13','23'],['13','29'],['13','40'],]#['5','7'],['7','6'],['10','4'],['10','70']
    for d in D:
        if folder==d[0] and box==d[1]:
            return True
    return False


def main():
    
    # if Path('maliciousurl_tool_data.json').exists():
    #     maliciousurl_tool_data = json.load(open('maliciousurl_tool_data.json'))
    #     maliciousurl_tool = MaliciousURLTool(maliciousurl_tool_data)
    # else:
    #     maliciousurl_tool = MaliciousURLTool([json.load(open(f'{folder}/data.json'))['base'] for folder in Path('/data1/lokesh/combineddata').iterdir() if json.load(open(f'{folder}/status.json'))[0] == 'appropriate'])

    for folder in tqdm(sorted([folder for folder in Path('/data1/lokesh/combineddata').iterdir()], key=lambda x: int(x.name))[:20]):
        # print(f'{folder}/{folder.name}/answers.json')
        # if Path(f'{folder}/answers.json').exists():
        #     continue
        if int(folder.name) < 55:
            continue
        
        
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
                
                
                # messages = []
                # messages += [
                #     {"role": "user", "content": [
                #         {"image": f"temp_each_box/{img_num}/{box}.jpg"},
                #         {"image": f"{folder}/screenshots/{box}.jpg"},
                #         {"image": f"{folder}/box_crops/{box}.jpg"},
                #         {"text":  f"Box is highlighted with {sommer.color_list[f'{img_num}_{box}']} color."
                #         f"Center of the highlighted box is {json_data[box]['x']+json_data[box]['width']/2} and {modified_y+json_data[box]['height']/2} in raw pixel space. Image Size is {sommer.outputs[img_num].width} and {sommer.outputs[img_num].height}."
                #         f"Normalized Center of the highlighted box is {(json_data[box]['x']+json_data[box]['width']/2)/sommer.outputs[img_num].width} and {(modified_y+json_data[box]['height']/2)/sommer.outputs[img_num].height}."
                #         f"URL of first image is {json_data['base']}. URL of second image is {json_data[box]['url']}"
                #         f"Derived features from URLs of both images are {deriveUrlFeatures(url1,url2)}."
                #         }
                #     ]}
                # ]
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
                max_tokens=4096
            )
                

                # try:
                #     response_plain_text = ''
                #     for ret_messages in agent.run(messages):
                #         # `ret_messages` will contain all subsequent messages, consisting of interleaved assistant messages and tool responses
                #         response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
                # except Exception as e:
                #     continue
                # final_messages = list(agent.run(messages))[0]
                # print(final_messages)
                # answer = extract_answer(response_plain_text)
                # answers_dict[key] = answer
                message = response.choices[0].message
                response_text = message.content
                if hasattr(message, "reasoning") and message.reasoning:
                    reasoning = message.reasoning
                    response_text += f"\n\n[Reasoning]: {reasoning}"
                answers_dict[f"{img_num}_{box}"] = response_text
                pbar.update(1)
            
        shutil.rmtree(f'./temp_each_box')
        Path(f'gemma_vlm1_reasoning_wobf').mkdir(parents=True, exist_ok=True)
        with open(f'gemma_vlm1_reasoning_wobf/{folder.name}.json', 'w') as f:
            json.dump(answers_dict, f, indent=4)    

        #Save maliciousurl_tool data
        # with open('maliciousurl_tool_data.json', 'w') as f:    
        #     json.dump(maliciousurl_tool.url_list, f, indent=4)  
        #     json.dump(maliciousurl_tool.url_list, f, indent=4)  

        
        # for key in tqdm([key for key in json_data.keys() if key != 'base']):
        #     if "error: " in json_data[key]["url"]:
        #         continue
        #     base_img_num = get_img_num(json_data[key]["y"], json_data[key]["height"], scroll_info)
            
        #     modified_y = json_data[key]["y"] - sum(scroll_info[:base_img_num])

        #     url1 = json_data['base']
        #     url2 = json_data[key]['url']
            
            
        #     if not Path(f"{folder}/screenshots/{key}.jpg").exists():
        #         continue
        #     messages = []
        #     messages += [
        #         {"role": "user", "content": [
        #             {"image": f"temp_each_box/{curr_folder_number}/{key}.jpg"},
        #             {"image": f"{folder}/screenshots/{key}.jpg"},
        #             {"text": f"Center of the highlighted box is {json_data[key]['x']+json_data[key]['width']/2} and {modified_y+json_data[key]['height']/2}."
        #             f"URL of first image is {json_data['base']}. URL of second image is {json_data[key]['url']}"
        #             f"Derived features from URLs of both images are {deriveUrlFeatures(url1,url2,maliciousurl_tool)}."
        #             }
        #         ]}
        #     ]
            
        #     response_plain_text = ''
        #     for ret_messages in agent.run(messages):
        #         # `ret_messages` will contain all subsequent messages, consisting of interleaved assistant messages and tool responses
        #         response_plain_text = multimodal_typewriter_print(ret_messages, response_plain_text)
        #     # final_messages = list(agent.run(messages))[0]
        #     # print(final_messages)
        #     # answer = extract_answer(response_plain_text)
        #     # answers_dict[key] = answer
        #     answers_dict[key] = response_plain_text
        
        # shutil.rmtree(f'./temp_each_box/{curr_folder_number}')
        # with open(f'{folder}/answers_qwen35_reasoning.json', 'w') as f:
        #     json.dump(answers_dict, f)    

        # #Save maliciousurl_tool data
        # with open('maliciousurl_tool_data.json', 'w') as f:    
        #     json.dump(maliciousurl_tool.url_list, f, indent=4)        

if __name__ == "__main__":
    main()      