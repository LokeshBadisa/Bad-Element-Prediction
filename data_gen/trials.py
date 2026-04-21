# from pathlib import Path    
# import json
# L = []
# for folder in sorted(Path('/data1/lokesh/data').iterdir(),key=lambda x: int(x.name)):    
#     if (folder / 'data.json').exists() and (folder / 'quality_check/image.json').exists():
#         image_json = json.load((folder / 'quality_check/image.json').open())
#         if image_json[0] != 'complete':
#             continue
#         jsondata = json.load((folder / 'data.json').open())
#         if '-1' not in jsondata and len(list(jsondata.keys())) > 1:
#             L.append(folder.name)   

# print(L,len(L))

from preprocess_utils import *
from tqdm import tqdm
from pathlib import Path
import json


def main():
    folders = sorted(list(Path('/data1/lokesh/tranco_data/bep/data_gen/data').iterdir()),key=lambda x: int(x.name))
    box_count_tranco = json.load(open('box_count_tranco.json', 'r')) if Path('box_count_tranco.json').exists() else {}
    results = {}

    for folder in tqdm(folders):
        if folder.name in box_count_tranco:
            results[folder.name] = box_count_tranco[folder.name]
            continue
        try:
            print(folder.name)
            sommer = SoM(
                folder / 'base_screenshots',
                folder / 'data.json',
                folder / 'base_screenshots/metadata.json'
            )
            S = set()
            for k, v in sommer.boxes_in_image.items():
                S.update(v)
            results[folder.name] = len(S)
            with open('box_count_tranco.json', 'w') as f:
                json.dump(results, f)
        except Exception as e:
            print(f"[ERROR] {folder.name}: {e}", flush=True)

    


if __name__ == '__main__':
    main()