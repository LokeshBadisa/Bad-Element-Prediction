from pathlib import Path    
import json
L = []
for folder in sorted(Path('/data1/lokesh/data').iterdir(),key=lambda x: int(x.name)):    
    if (folder / 'data.json').exists() and (folder / 'quality_check/image.json').exists():
        image_json = json.load((folder / 'quality_check/image.json').open())
        if image_json[0] != 'complete':
            continue
        jsondata = json.load((folder / 'data.json').open())
        if '-1' not in jsondata and len(list(jsondata.keys())) > 1:
            L.append(folder.name)   

print(L,len(L))