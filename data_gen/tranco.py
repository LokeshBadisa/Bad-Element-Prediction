import asyncio
import json
import polars as pl
from tqdm.asyncio import tqdm_asyncio
from playwright.async_api import async_playwright
from argparse import ArgumentParser
from pathlib import Path

CONCURRENCY = 100   


async def load_page(semaphore, context, url):
    async with semaphore:
        page = await context.new_page()
        try:
            await page.goto('https://' + url+'/', timeout=10000)
            final_url = page.url
            await page.close()
            return url, final_url
        except Exception as e:
            await page.close()
            return url, None        
            


async def main():
    # Load URLs from CSV
    df = pl.read_csv("top-1m.csv")
    if not Path("tranco_1M.json").exists():
        json.dump([], open("tranco_1M.json", "w"), indent=4)
    processed_urls = set(json.load(open("tranco_1M.json", "r")))
    urls = df.select(pl.col("url")).to_series().to_list()

    urls = [u for u in urls if u not in processed_urls]

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # Shared context
        context = await browser.new_context()

        tasks = [
            load_page(semaphore, context, url)
            for url in urls
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Processing URLs", total=len(urls))

        await context.close()
        await browser.close()

    
    D = {url: final_url for url, final_url in results if final_url is not None}
    
    with open(f"Tranco/tranco_results.json", "w") as f:
        json.dump(D, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
