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

## Get URLs
### Benign URLs

1. Download Tranco top 1M dataset from https://tranco-list.eu/ and place the `top-1m.csv` file in the same directory as `tranco.py`.
2. `python3 tranco.py` to get URLs from Tranco domains.


Run the main data collection script.
```bash
python3 main.py
```
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

## Future Directions
1. Data structure and algorithm improvements:
   1. Efficient tree matching
   2. Efficient node elimination
2. Javascript enhancements to the whole algorithm
   1. Extracting elements (reducing redundant elements and adding elements which are missed by the current implementation)
   2. Script injections to playwright to improve robustness
   3. Implement to wait for the page till all images are loaded
3. Better heuristics for node elimination
4. Better handling of edge cases (e.g. pages with infinite scroll, pages with dynamic content, popups, changing homepages each time opened(with different products to showcase), etc.).
5. Instead of collecting box information only once, collect it for each scroll position. This will help in better handling of navbar and fixed position elements(like ask a question button in bottom left).