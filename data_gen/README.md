Setup

```bash
conda create -n datacoll python=3.11 -y
conda activate datacoll
pip install -r requirements.txt
playwright install
```

Create a .env file as:
```
VT_KEY=your_virustotal_api_key
```
Get your VirusTotal API key from https://www.virustotal.com/gui/my-apikey

Generate URLs file from Gigasheet dataset.
```bash
python3 url_data_getter.py
```

Run the main data collection script.
```bash
python3 main.py
```

Future Directions:
1. Data structure and algorithm improvements:
   1. Efficient tree matching
   2. Efficient node elimination
2. Javascript enhancements to the whole algorithm
   1. Extracting elements (reducing redundant elements and adding elements which are missed by the current implementation)
   2. Script injections to playwright to improve robustness
   3. Implement to wait for the page till all images are loaded
3. Better heuristics for node elimination
4. Better handling of edge cases (e.g. pages with infinite scroll, pages with dynamic content, popups, changing homepages each time opened(with different products to showcase), etc.).