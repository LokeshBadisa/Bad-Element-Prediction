import json
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from fast_ssim import ssim
from tqdm import tqdm
from urllib.parse import urlparse
from preprocess_utils import *

BASE_DIR = '/data1/lokesh/v2_zip_openphish_usable'

# for folder in Path(BASE_DIR).iterdir():
#     base_url = json.load(open(folder/'data.json'))['base']
#     if 'github' in base_url or 'clone' in base_url or 'gitlab' in base_url or 'gitbook' in base_url:
#         shutil.rmtree(folder)

# # URL deduplication
# url_set = set()
# duplicate_count = 0
# for folder in Path(BASE_DIR).iterdir():
#     base_url = json.load(open(folder/'data.json'))['base']
#     if base_url not in url_set:
#         url_set.add(base_url)
#     else:
#         shutil.rmtree(folder)
#         duplicate_count += 1
# print(f'URL based duplicates: {duplicate_count}')

# # Domain-based deduplication
# domain_set = {}
# domain_duplicate_count = 0
# for folder in Path(BASE_DIR).iterdir():
#     base_url = json.load(open(folder/'data.json'))['base']
#     domain = urlparse(base_url).netloc
#     if domain not in domain_set:
#         domain_set[domain] = base_url 
#     else:
#         shutil.rmtree(folder)
#         domain_duplicate_count += 1

# print(f'Domain-based duplicates: {domain_duplicate_count}')

#Image-based deduplication
# folder_list = json.load(open('noclone_shubho_till_may20.json'))
# folder_list = [folder.name for folder in Path('/data1/lokesh/v2_zip_openphish_usable').iterdir() if int(folder.name)>=64761]
# image_dict = {}
# ssim_dict = json.load(open('ssim_clusters_may27.json'))
# for folder in tqdm([Path(BASE_DIR)/f'{folder}' for folder in folder_list]):    
#     img = np.array(Image.open(folder/f'screenshot.jpg'))
#     found_cluster = False
#     for cluster_id,cluster_img in reversed(image_dict.items()):
#         if img.shape == cluster_img.shape:
#             score = ssim(img, cluster_img)
#             if np.count_nonzero(img - cluster_img) <= 25 or score > 0.99:
#                 if cluster_id not in ssim_dict:
#                     ssim_dict[cluster_id] = [cluster_id]
#                 ssim_dict[cluster_id].append((folder.name,score))
#                 found_cluster = True
#                 break
#     if not found_cluster:
#         image_dict[folder.name] = img

# with open('ssim_clusters_may28.json', 'w') as f:
#     json.dump(ssim_dict, f, indent=4)

ssim_dict = json.load(open('ssim_clusters_may28.json'))
retain_dict = {}
for key in tqdm(ssim_dict):
    if len(ssim_dict[key]) > 1:
        # Path(f'ssim_checking/{key}').mkdir(parents=True, exist_ok=True)
        try:
            sommer = SoM(f'{BASE_DIR}/{key}/base_screenshots', Path(f'{BASE_DIR}/{key}/data.json'),\
                        f'{BASE_DIR}/{key}/base_screenshots/metadata.json')
        except:
            continue
        max_count = sommer.unique_box_count
        max_count_folder = key
        for holder in ssim_dict[key][1:]:
            folder_name = holder[0] if isinstance(holder, tuple) or isinstance(holder, list) else holder
            try:
                sommer = SoM(f'{BASE_DIR}/{folder_name}/base_screenshots', Path(f'{BASE_DIR}/{folder_name}/data.json'),\
                        f'{BASE_DIR}/{folder_name}/base_screenshots/metadata.json')
            except Exception as e:
                print(f"Error occurred while processing {folder_name}: {e}")
                continue
            
            if sommer.unique_box_count>max_count:
                max_count = sommer.unique_box_count
                max_count_folder = folder_name
            # shutil.copy((Path(BASE_DIR)/f'{folder_name}/screenshot.jpg'), f'ssim_checking/{key}/{folder_name}.jpg')
        retain_dict[key] = max_count_folder
# print(ssim_dict)
with open('retain_dict_may28.json', 'w') as f:
    json.dump(retain_dict, f, indent=4)