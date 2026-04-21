from tqdm import tqdm
import json
from pathlib import Path
from preprocess_utils import SoM
import requests
from dotenv import load_dotenv
import os
import time

load_dotenv()
VT_KEY = os.getenv("VT_KEY")


def scanUrl(url):    
    headers = {
        "x-apikey": VT_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response =  requests.post(
        "https://www.virustotal.com/api/v3/urls",
        headers=headers,
        data={"url": url}
    )
    return response.json()

def getAnalysis(report_id):
    url = f"https://www.virustotal.com/api/v3/analyses/{report_id}"
    headers = {"accept": "application/json", "x-apikey": VT_KEY}    
    response =  requests.get(url, headers=headers, timeout=10)
    return response.json()

def isUrlMalicious(sliding_window, url):       
    try: 
        while len(sliding_window) >= 4 and current_time - sliding_window[0] < 60:
            time.sleep(1)
            current_time = time.time()
        while len(sliding_window) >= 4:
            sliding_window.pop(0)
        response =  scanUrl(url)
        report_id = response['data']['id']
    except Exception as e:
        print(VT_KEY,response)
    #     return 'unknown'
    response =  getAnalysis(report_id)
    return True if response['data']['attributes']['stats']['malicious'] > 0 else False

class MaliciousURLTool:
    def __init__(self,url_list):
        #Counter example for why we use url instead of domain: 
        #https://accounts.google.com/v3/signin/identifier?continue=https://sites.google.com/view/bhds2sdadasd/inicio&dsh=S-1861035836:1771233705536992&followup=https://sites.google.com/view/bhds2sdadasd/inicio&ifkv=ASfE1-qnhQ2GLicynTYpejuzD4_8mEOCcHSQXu2IMz7UveC0G8FqSk6DJ4VOampxCFJP96MSKRxL5Q&osid=1&passive=1209600&flowName=WebLiteSignIn&flowEntry=ServiceLogin
        self.url_list = {url: "malicious" for url in url_list}
        self.sliding_window = []

    def is_malicious(self, url):
        if len(self.sliding_window) >= 499:
            raise Exception("Daily Quota exceeded for VirusTotal API")
        if url in self.url_list:
            return self.url_list[url]
        else:
            result = isUrlMalicious(self.sliding_window, url)
            # if result == 'unknown':
            #     self.url_list[url] = 'unknown'
            # else:
            self.url_list[url] = "malicious" if result else "benign"
            return self.url_list[url]


if Path('maliciousurl_tool_data.json').exists():
    maliciousurl_tool_data = json.load(open('maliciousurl_tool_data.json'))
    maliciousurl_tool = MaliciousURLTool([])
    maliciousurl_tool.url_list = maliciousurl_tool_data
else:
    maliciousurl_tool = MaliciousURLTool([json.load(open(f'{folder}/data.json'))['base'] for folder in Path('/data1/lokesh/combineddata').iterdir() if json.load(open(f'{folder}/status.json'))[0] == 'appropriate'])


for folder in tqdm(sorted([folder for folder in Path('/data1/lokesh/combineddata').iterdir() if json.load(open(f'{folder}/status.json'))[0] == 'appropriate'])):
    # print(f'{folder}/{folder.name}/answers.json')
    # if Path(f'{folder}/answers.json').exists():
    #     continue
    json_data = json.load(open(f'{folder}/data.json'))    
    scroll_info = json.load(open(f'{folder}/base_screenshots/metadata.json'))['scroll_steps']
    
    sommer = SoM(f'{folder}/base_screenshots', Path(f'{folder}/data.json'), f'{folder}/base_screenshots/metadata.json')
    curr_folder_number = folder.name

    #make a tqdm counter for the boxes in the image
    pbar = tqdm(total=sum([len(boxes) for boxes in sommer.boxes_in_image.values()]))
    for img_num, boxes_list in sommer.boxes_in_image.items():
        for box in boxes_list:
            if "error: " in json_data[box]["url"]:
                continue    

            modified_y = json_data[box]["y"] - sum(scroll_info[:img_num])

            url1 = json_data['base']
            url2 = json_data[box]['url']            
            maliciousurl_tool.is_malicious(url2)
            
            
            pbar.update(1)
        
    with open('maliciousurl_tool_data.json', 'w') as f:    
        json.dump(maliciousurl_tool.url_list, f, indent=4)  