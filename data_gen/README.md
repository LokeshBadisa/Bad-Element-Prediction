## Table of Contents


- [Table of Contents](#table-of-contents)
- [Setup](#setup)
- [Get URLs](#get-urls)
  - [Benign URLs](#benign-urls)
- [Label Generation](#label-generation)
- [Convert to ShareGPT format](#convert-to-sharegpt-format)
- [Future Directions](#future-directions)

## Setup

Please see requirements.txt for task wise dependencies. 
```bash
conda create -n datacoll python=3.11 -y
conda activate datacoll
pip install -r requirements.txt
playwright install
pip install flash-attn --no-build-isolation --no-cache-dir
```

Create a .env file as:
```
VT_KEY=your_virustotal_api_key
```
Get your VirusTotal API key from https://www.virustotal.com/gui/my-apikey

## Step 1: Get URLs
### 1.1 Benign URLs

1. Download Tranco top 1M dataset from https://tranco-list.eu/ and place the `top-1m.csv` file in the same directory as `tranco.py`.
2. `python3 tranco.py` to get URLs from Tranco domains.

Instead of running `tranco.py`, you can download the output from [Google Drive](https://drive.google.com/file/d/1g2eUkw15HZBA_aX7b7PkJAuqpR4rc7ra/view?usp=sharing). This has 325K domain-URL pairs from Tranco dataset.


### 1.2 Malicious URLs
To generate malicious URLs data, we assume you have a list of malicious URLs in a json file with the following format:
```json
[
   "url1",
   "url2",
   ...
]
```

To deduplicate URLs:
```bash
python3 url_deduplication.py --input malicious_urls.json --output malicious_urls_dedup.json
```

## Step 2: Data Collection and Cleaning

Run the main data collection script.
```bash
python3 main.py
```

After data collection finishes, run:
```bash
python3 data_cleaning.py
```
to remove any URLs which have issues in data collection (e.g. pages which are not loading, pages which are showing some error, zero elements and >100 elements). 

## Label Generation
```
sh vllm_serve.sh
python3 quality_check_images.py
python3 quality_check_boxes.py
python3 labelling.py
```
**Major Note: Qwen 3 VL 8B is used instead of 32B variant.**

## Convert to ShareGPT format
```
python3 sharegpt.py
```

### File Descriptions
1. `labelling_utils.py`: Functions for auxiliary information about URL. Used in getting base reasoning from GPT and also in verifier prompts.
2. `labelling.py`: Main script for generating base reasoning
3. `main.py`: Main script for data scraping
4. `preprocess_utils.py`: Function(`SoM`) for saving boxes highlighted as input to GPT and verifier
5. `prompts.py`: Prompts
6. `quality_check_images.py`: Script to check if the redirected page screenshots have any information or not
7. `sharegpt.py`: Script to convert the generated labels to sharegpt format
8. `tranco.py`: Script to get URLs from Tranco dataset
9. `utils.py`: Utility functions for data collection and labelling
10. `verifier.py`: Gemma based verifier to verify GPT generated reasoning
11. `virustotal.py`: 

## TODO.md
1. The logic of not clicking on buttons where the crop in any of the scroll screenshots(`base_screenshots`) should be implemented in the main data collection script instead of just in the `preprocess_utils.py` `SoM` function. 

File meanings:
1. tranco_usable.json : output of qci_async.py on tranco
2. shubho_usable.json : output of qci_async.py on openphish_v1 
3. bus_json: folder of box usability shubho
4. but_json: folder of box usability tranco


## Future Directions
1. Data structure and algorithm improvements(Implementation):
   1. Efficient tree matching
   2. Efficient node elimination
2. Javascript enhancements to the whole algorithm(Implementation):
   1. Extracting elements (reducing redundant elements and adding elements which are missed by the current implementation)
   2. Script injections to playwright to improve robustness
   3. Implement to wait for the page till all images are loaded
3. Better heuristics for node elimination(Research)
4. Better handling of edge cases (e.g. pages with infinite scroll, pages with dynamic content, popups, changing homepages each time opened(with different products to showcase), etc.). (Implementation)
5. Instead of collecting box information only once, collect it for each scroll position. This will help in better handling of navbar and fixed position elements(like ask a question button in bottom left). (Implementation)
6. Image based deduplication while collecting data instead of post-data collection. (Implementation)