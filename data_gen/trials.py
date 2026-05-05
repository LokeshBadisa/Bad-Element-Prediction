from preprocess_utils import SoM
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def count_active_boxes(folder):
    try:
        sommer = SoM(
            Path(f'/data1/lokesh/tranco_data/bep/data_gen/data/{folder}') / "base_screenshots",
            Path(f'/data1/lokesh/tranco_data/bep/data_gen/data/{folder}') / "data.json",
            Path(f'/data1/lokesh/tranco_data/bep/data_gen/data/{folder}') / "base_screenshots/metadata.json"
        )
        return len(sommer.get_active_boxes())
    except Exception as e:
        print(f"[WARN] SoM failed for {folder}: {e}")
        return 0

# folders = sorted(Path("/data1/lokesh/shubho/").iterdir(), key=lambda x: int(x.stem))
folders = json.load(open('tranco_usable.json'))
total = 0

with ThreadPoolExecutor(max_workers=12) as executor:  # tune this
    futures = {executor.submit(count_active_boxes, folder): folder for folder in folders}
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            total += future.result(timeout=60)  # per-task timeout
        except Exception as e:
            folder = futures[future]
            print(f"[ERROR] {folder.stem}: {e}")

print(f"Total: {total}")