from utils import *
from urllib.parse import urlparse
import pandas as pd
import asyncio
import time
from playwright_stealth import Stealth# type: ignore

MAX_CONCURRENT_URLS = 6

CONTEXT_POOL_SIZE = MAX_CONCURRENT_URLS

# If no progress (no URL marked done) for this many seconds, cancel pending tasks
INACTIVITY_TIMEOUT = 1200



async def process_one_url(number, url, pool, url_sem, progress):   
    async with url_sem:
        context = await pool.acquire()
        try:
            # Stage 1: Collect single link data with 5-min timeout
            try:
                # Run collector normally; cancellation will be handled by the inactivity watchdog
                result = await single_link_collector(number, url, context)
            except asyncio.CancelledError:
                print(f"Timeout exceeded (inactivity) in single_link_collector for URL {number}")
                Path(f"{save_dir}/{number}").mkdir(parents=True, exist_ok=True)
                with open(f"{save_dir}/{number}/error.log", "w") as f:
                    f.write("Timeout exceeded (inactivity) in single_link_collector stage")
                return

            if result is None:
                # print(f"Error unpacking result for URL {number}: {e}")
                Path(f"{save_dir}/{number}").mkdir(parents=True, exist_ok=True)
                with open(f"{save_dir}/{number}/error.log", "w") as f:
                    f.write("single_link_collector returned None")
                return
            
            try:
                number, new_roots, LargeMatching, node_storage, isomorphs, url_dict, download_dict = result
            except Exception as e:
                # print(f"Error unpacking result for URL {number}: {e}")
                Path(f"{save_dir}/{number}").mkdir(parents=True, exist_ok=True)
                with open(f"{save_dir}/{number}/error.log", "w") as f:
                    f.write(f"Unpacking error: {e}")
                return

            if '-1' in url_dict.keys():
                #Save url_dict as json
                with open(f"{save_dir}/{number}/data.json", "w") as f:
                    json.dump(url_dict, f)
                print(f"Skipping URL {number} due to -1 in url_dict")
                return
                
            # --- Stage 2 (synchronous / CPU-bound) ---
            newnew_roots, _, _ = root_level_isomorphism(
                new_roots, LargeMatching, node_storage,
                isomorphs, f"{save_dir}/{number}",
            )
            
            if '-1' in url_dict.keys():
                #Save url_dict as json
                with open(f"{save_dir}/{number}/data.json", "w") as f:
                    json.dump(url_dict, f)
                return

            # --- Stage 3 (reuses the SAME context — shared cache) ---
            try:
                # Run collect_data normally; cancellation will be handled by watchdog
                await collect_data(
                    number, newnew_roots,
                    url_dict, download_dict,
                    url, f"{save_dir}/{number}", context,
                )
            except asyncio.CancelledError:
                print(f"Timeout exceeded (inactivity) in collect_data for URL {number}")
                Path(f"{save_dir}/{number}").mkdir(parents=True, exist_ok=True)
                with open(f"{save_dir}/{number}/error.log", "a") as f:
                    f.write("\nTimeout exceeded (inactivity) in collect_data stage")
                return

            progress["success"] += 1

        except Exception as e:
            Path(f"{save_dir}/{number}").mkdir(parents=True, exist_ok=True)
            with open(f"{save_dir}/{number}/error.log", "w") as f:
                f.write(repr(e))

        finally:
            # ALWAYS update done + release context
            progress["done"] += 1
            await pool.release(context)


BATCH_SIZE = 50

async def main(url_list):
    # Filter url_list first
    to_process = [
        (i, url) for i, url in enumerate(url_list) 
        if not Path(f"{save_dir}/{i}/data.json").exists() and not Path(f"{save_dir}/{i}/error.log").exists()
    ]
    
    total_to_process = len(to_process)
    print(f"Total URLs to process: {total_to_process}")

    for batch_start in range(0, total_to_process, BATCH_SIZE):
        batch = to_process[batch_start:batch_start + BATCH_SIZE]
        print(f"Starting batch {batch_start // BATCH_SIZE + 1} ({len(batch)} URLs)...")
        
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(headless=True,
                                              args=["--disable-remote-fonts"])
            
            pool = ContextPool(browser, pool_size=CONTEXT_POOL_SIZE)
            url_sem = asyncio.Semaphore(MAX_CONCURRENT_URLS)
            progress = {"done": 0, "success": 0, "total": len(batch)}

            try:
                # Create real asyncio Tasks so we can cancel them via watchdog
                tasks = [
                    asyncio.create_task(process_one_url(i, url, pool, url_sem, progress))
                    for i, url in batch
                ]

                async def inactivity_watchdog(progress, tasks, timeout):
                    last_done = progress.get("done", 0)
                    last_time = time.monotonic()
                    while True:
                        await asyncio.sleep(1)
                        # progress changed -> reset timer
                        if progress.get("done", 0) != last_done:
                            last_done = progress.get("done", 0)
                            last_time = time.monotonic()
                        # no progress for `timeout` seconds -> cancel pending tasks
                        elif time.monotonic() - last_time > timeout:
                            pending = [t for t in tasks if not t.done()]
                            if pending:
                                print(f"No progress for {timeout}s — cancelling {len(pending)} pending tasks")
                                for t in pending:
                                    t.cancel()
                            break
                        # all done -> exit
                        if all(t.done() for t in tasks):
                            break

                watchdog = asyncio.create_task(inactivity_watchdog(progress, tasks, INACTIVITY_TIMEOUT))

                # Use as_completed so we can show a progress bar and handle cancellations
                for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Batch {batch_start // BATCH_SIZE + 1}"):
                    try:
                        await coro
                    except asyncio.CancelledError:
                        # Task was cancelled by watchdog due to inactivity
                        pass
                    except Exception as e:
                        print(f"Task error: {e}")

                # Wait for watchdog to finish (it may have been the thing that cancelled tasks)
                await watchdog

            except Exception as e:
                print(f"Batch {batch_start // BATCH_SIZE + 1} failed with error: {e}")
            finally:
                # --- Tear down every pooled context before closing the browser ---
                await pool.close_all()
                await browser.close()



if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Data collection for BEP")
    # parser.add_argument("--dataset", type=str, default="phishtank" )
    # parser.add_argument("--data_path", type=str)
    # args = parser.parse_args()
    # if args.dataset == "phishtank":        
    #     phishtank = json.load(open(args.data_path))
    #     phishtank = clean_phishtank(phishtank)
    #     #print(phishtank.values())
    #     url_list = list(phishtank.values())
    # asyncio.run(main(list(json.load(open("/data1/lokesh/blp/data-annotation/tranco_1M.json")).values())[:500]))
    # data = json.load(open('phishing_feed_30_days.json'))
    # df = pd.DataFrame(data)
    # asyncio.run(main(list(df['Url'])))
    data = json.load(open('sampled_30k.json'))
    asyncio.run(main(list(data)))