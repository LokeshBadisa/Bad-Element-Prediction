import json
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from fast_ssim import ssim
from tqdm import tqdm
from urllib.parse import urlparse
from preprocess_utils import *

url_list = [url.strip() for url in open('feed_May31_unique.txt').readlines()]
url_list = list(set(url_list))  # initial deduplication
removal_list = []
git_removal = 0
for base_url in url_list:    
    if 'clone' in base_url.lower() or 't-mobile' in base_url.lower():
        removal_list.append(base_url)
        git_removal += 1

url_list = [url for url in url_list if url not in removal_list]
print(f'Clone-based removals: {git_removal}')

# URL deduplication
url_set = set()
duplicate_count = 0
removal_list = []
for base_url in url_list:
    if base_url not in url_set:
        url_set.add(base_url)
    else:
        removal_list.append(base_url)
        duplicate_count += 1
print(f'URL based duplicates: {duplicate_count}')
url_list = [url for url in url_list if url not in removal_list]

# Domain-based deduplication
domain_set = {}
domain_duplicate_count = 0
removal_list = []
for base_url in url_list:
    domain = urlparse(base_url).netloc
    if domain not in domain_set:
        domain_set[domain] = base_url 
    else:
        removal_list.append(base_url)
        domain_duplicate_count += 1

print(f'Domain-based duplicates: {domain_duplicate_count}')
url_list = [url for url in url_list if url not in removal_list]

with open('feed_May31_deduplicated.txt', 'w') as f:
    for url in url_list:
        f.write(f"{url}\n")

# import asyncio
# import json
# from pathlib import Path
# from tqdm.asyncio import tqdm_asyncio
# import json
# import shutil
# import numpy as np
# from PIL import Image
# from pathlib import Path
# from fast_ssim import ssim
# from tqdm import tqdm
# from urllib.parse import urlparse
# from preprocess_utils import *

# MAX_CONCURRENCY = 10  # tune based on I/O vs CPU profile
# semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
# BASE_DIR = '/data1/lokesh/v2_zip_openphish_usable'

# async def process_cluster(key, holders, base_dir):
#     async with semaphore:
#         loop = asyncio.get_running_loop()

#         Path(f'ssim_checking/{key}').mkdir(parents=True, exist_ok=True)

#         # SoM is likely CPU/IO-bound — run in executor to avoid blocking event loop
#         def compute_best_folder():
#             sommer = SoM(
#                 f'{base_dir}/{key}/base_screenshots',
#                 Path(f'{base_dir}/{key}/data.json'),
#                 f'{base_dir}/{key}/base_screenshots/metadata.json'
#             )
#             max_count = sommer.unique_box_count
#             max_count_folder = key

#             for holder in holders[1:]:
#                 folder_name = holder[0] if isinstance(holder, (tuple, list)) else holder
#                 sommer = SoM(
#                     f'{base_dir}/{folder_name}/base_screenshots',
#                     Path(f'{base_dir}/{folder_name}/data.json'),
#                     f'{base_dir}/{folder_name}/base_screenshots/metadata.json'
#                 )
#                 if sommer.unique_box_count > max_count:
#                     max_count = sommer.unique_box_count
#                     max_count_folder = folder_name

#             return max_count_folder

#         best_folder = await loop.run_in_executor(None, compute_best_folder)
#         return key, best_folder


# async def main():
#     ssim_dict = json.load(open('ssim_clusters_may20.json'))

#     tasks = [
#         process_cluster(key, holders, BASE_DIR)
#         for key, holders in ssim_dict.items()
#         if len(holders) > 1
#     ]

#     results = await tqdm_asyncio.gather(*tasks, desc="Processing clusters")

#     retain_dict = {key: folder for key, folder in results}
#     return retain_dict


# retain_dict = asyncio.run(main())
# with open('retain_dict_may20.json', 'w') as f:
#     json.dump(retain_dict, f, indent=4)