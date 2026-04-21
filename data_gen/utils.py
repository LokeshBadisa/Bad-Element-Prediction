import asyncio
import cv2
import os
from PIL import Image
import random
import json
import copy
import numpy as np
from fast_ssim import ssim# type: ignore
from urllib.parse import urlparse
import time
from pathlib import Path
from tqdm import tqdm
import hashlib
import requests
from dotenv import load_dotenv
from io import BytesIO
from playwright.async_api import Error as PlaywrightError# type: ignore
from collections import defaultdict
from playwright.async_api import async_playwright# type: ignore

load_dotenv()
VT_KEY = os.getenv("VT_KEY")
save_dir = 'openphishdata'


############################################
############ Data collection utils #########
############################################
# ---------------------------------------------------------------------------
# Context Pool — reusable browser contexts with a hard concurrency cap
# ---------------------------------------------------------------------------
class ContextPool:
    """
    Maintains a fixed-size pool of reusable Playwright browser contexts.

    * `acquire()` returns an idle context (or creates one if the pool isn't
      at capacity yet).  The call blocks if all contexts are in use.
    * `release(ctx)` returns a context to the pool after clearing cookies
      and closing stale pages so the next user starts with a clean slate.
    * `close_all()` tears down every context in the pool (call once at
      shutdown).

    Using a pool avoids the two main problems in the original code:
      1. Context leak — `single_link_collector` never closed its context.
      2. Context-per-click — `take_screenshot` opened (and closed) a brand-
         new context for every single click, paying full DNS / TLS / nav
         cost each time.
    """

    def __init__(self, browser, pool_size=6):
        self._browser = browser
        self._sem = asyncio.Semaphore(pool_size)
        self._idle: asyncio.Queue = asyncio.Queue()
        self._all: list = []          # track every context for shutdown
        self._pool_size = pool_size

    async def _create_context(self):
        """Spin up a fresh context with the standard viewport & route."""
        ctx = await self._browser.new_context(
            ignore_https_errors=True,
            viewport={"width": 1920, "height": 1080},
        )
        await ctx.route("**/*", route_handler)
        self._all.append(ctx)
        return ctx

    async def acquire(self):
        """Get a ready-to-use context (waits if pool is exhausted)."""
        await self._sem.acquire()          # blocks when all contexts busy
        if not self._idle.empty():
            return self._idle.get_nowait()
        return await self._create_context()

    async def release(self, ctx):
        """Return a context to the pool after resetting it."""
        try:
            if not self._browser.is_connected():
                raise Exception("Browser disconnected")
            
            # Close every page except the blank first one
            for page in ctx.pages[1:]:
                try:
                    await page.close()
                except Exception:
                    pass
            # If the first page is still around, navigate it to blank
            if ctx.pages:
                try:
                    await ctx.pages[0].goto("about:blank", timeout=5000)
                except Exception:
                    pass
            await ctx.clear_cookies()
        except Exception:
            # Context is broken — replace it
            try:
                await ctx.close()
            except Exception:
                pass
            if ctx in self._all:
                self._all.remove(ctx)
            ctx = await self._create_context()

        self._idle.put_nowait(ctx)
        self._sem.release()

    async def close_all(self):
        """Shut down every context (call at program exit)."""
        for ctx in self._all:
            try:
                await ctx.close()
            except Exception:
                pass
        self._all.clear()
        # Drain the queue
        while not self._idle.empty():
            self._idle.get_nowait()


class BoxNode:
    def __init__(self, box, index):
        self.box = box
        self.isomorphs = {}
        self.height = 0
        self.idx = index
        self.children = []
        self.efficient_clicks = {}
        self.keep = True
        self.no_points_outside_children = False
        self.node_storage = {}
        self.LargeMatching = {}

    def __repr__(self):
        return f"Box{self.idx}({self.box})"
    
    def sample_points_outside_children(self):
        """Sample a point inside the box but outside all children.
        If no such point exists (element fully covered by interactable children),
        returns the center of the box as a fallback."""
        uncovered_point = has_empty_space(
            (self.box['x'], self.box['y'], self.box['width'], self.box['height']),
            [(child.box['x'], child.box['y'], child.box['width'], child.box['height']) for child in self.children]
        )
        # if uncovered_point and isinstance(uncovered_point, tuple):
        return uncovered_point
        
        # Fallback to center if fully covered
        # return (self.box['x'] + self.box['width'] / 2, self.box['y'] + self.box['height'] / 2)
    
    def any_area_left(self):
        if not self.children:
            return True
        return has_empty_space(
            (self.box['x'], self.box['y'], self.box['width'], self.box['height']),
            [(child.box['x'], child.box['y'], child.box['width'], child.box['height']) for child in self.children]
        ) is not None
    
    def get_img_num(self,y,scroll_info):    
            centers = [540+sum(scroll_info[:i]) for i in range(len(scroll_info)+1)]
            box_center = y 
            min_dist = float('inf')
            img_num = -1
            for i, center in enumerate(centers):
                dist = abs(center - box_center)
                if dist < min_dist:
                    min_dist = dist
                    img_num = i
            return img_num 
    
    def get_efficient_clicks(self,scroll_info):               
        
        self.LargeMatching, self.node_storage, self.isomorphs = match(self.children)
        
        self.efficient_clicks = {}
        for k in self.isomorphs.keys():
            node = find_node_by_idx(self, k)
            self.efficient_clicks.update(node.get_efficient_clicks(scroll_info))
        
        sampled_point = self.sample_points_outside_children()
        if sampled_point:
            self.efficient_clicks[self.idx] = (sampled_point, sum(scroll_info[:self.get_img_num(sampled_point[1], scroll_info)]))
        else:
            self.no_points_outside_children = True
            self.keep = False
            
        return self.efficient_clicks
    
    def get_keep_clicks(self, scroll_info):
        D = {}
        for child in self.children:
            D.update(child.get_keep_clicks(scroll_info))
        if self.keep:
            sampled_point = self.sample_points_outside_children()
            if sampled_point:
                D[self.idx] = (sampled_point, sum(scroll_info[:self.get_img_num(sampled_point[1], scroll_info)]))
        return D
    
    def get_keep_boxes(self):
        D = {}
        for child in self.children:
            D.update(child.get_keep_boxes())
        if self.keep:
            D[self.idx] = self.box
        return D


# ---------------------------------------------------------------------------
# Tree helpers (unchanged)
# ---------------------------------------------------------------------------
def match(children):
    LargeMatching = {}
    parent = {}
    node_storage = {}   

    for i,r1 in enumerate(children):            
        if parent.get(r1.idx) is None:
            parent[r1.idx] = r1.idx
            if node_storage.get(r1.idx) is None:
                node_storage[r1.idx] = r1
            for r2 in children[i+1:]:
                marker, matching = are_trees_identical(r1, r2)                    
                if marker:
                    parent[r2.idx] = r1.idx
                    LargeMatching[r2.idx] = r1.idx
                    LargeMatching.update(matching)

    for r1 in children:
        if LargeMatching.get(r1.idx) is None:
            LargeMatching[r1.idx] = r1.idx     
        if parent.get(r1.idx) is None:
            parent[r1.idx] = r1.idx  

    D = {}
    for k,v in parent.items():
        if D.get(v) is None:
            D[v] = []
        D[v].append(k)            
    
    return LargeMatching, node_storage, D

        
def is_inside(inner, outer):
    """Check if inner box is completely inside outer box"""
    inner_right = inner['x'] + inner['width']
    inner_bottom = inner['y'] + inner['height']
    outer_right = outer['x'] + outer['width']
    outer_bottom = outer['y'] + outer['height']
    
    return (inner['x'] >= outer['x'] and 
            inner['y'] >= outer['y'] and
            inner_right <= outer_right and
            inner_bottom <= outer_bottom)

def remove_mutual_containment_boxes(boxes):
    n = len(boxes)
    removed = set()

    for i in range(n):
        if i in removed:
            continue
        for j in range(i + 1, n):
            if j in removed:
                continue

            if is_inside(boxes[i], boxes[j]) and is_inside(boxes[j], boxes[i]):
                has_href_i = False
                has_href_j = False
                if boxes[i]['href'] and 'http' in boxes[i]['href']:
                    has_href_i = True
                if boxes[j]['href'] and 'http' in boxes[j]['href']:
                    has_href_j = True

                if has_href_i and not has_href_j:
                    removed.add(j)
                elif has_href_j and not has_href_i:
                    removed.add(i)
                    break
                else:
                    removed.add(j)

    return [box for idx, box in enumerate(boxes) if idx not in removed]

def build_bounding_box_tree(boxes):
    boxes = remove_mutual_containment_boxes(boxes)
    nodes = [BoxNode(box, i) for i, box in enumerate(boxes)]
    n = len(nodes)

    parent = [None] * n

    for i in range(n):
        best_parent = None
        best_area = float('inf')

        for j in range(n):
            if i == j:
                continue
            if is_inside(boxes[i], boxes[j]):
                area = boxes[j]['width'] * boxes[j]['height']
                if area < best_area:
                    best_parent = j
                    best_area = area

        if best_parent is not None:
            parent[i] = best_parent
            nodes[best_parent].children.append(nodes[i])

    roots = [nodes[i] for i in range(n) if parent[i] is None]

    return roots, n

def eliminate_inactive_nodes(roots):
    def helper(node, parent_children):
        if node.redirection == '':
            for child in node.children:
                parent_children.append(child)
        else:
            new_children = []
            for child in node.children:
                helper(child, new_children)
            node.children = new_children
            parent_children.append(node)
    new_roots = []
    for root in roots:
        helper(root, new_roots)
    return new_roots


def draw_boxes(imgname, roots, finalname, path, draw_false_keep=False):
    img = cv2.imread(imgname)
    for i,root in enumerate(roots):
        nodes_to_process = [root]
        while nodes_to_process:
            node = nodes_to_process.pop()
            ann = node.box
            if not draw_false_keep and not node.keep:
                continue
            cv2.rectangle(img,[int(ann['x']),int(ann['y'])],[int(ann['x']+ann['width']),int(ann['y']+ann['height'])],(random.randint(0,255), random.randint(0,255), random.randint(0,255)),2)
            cv2.putText(img, str(node.idx), (int(ann['x']), int(ann['y'])+20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 2)
            nodes_to_process.extend(node.children)
    cv2.imwrite(f'{path}/{finalname}_boxes.jpg', img)

def print_bounding_box_tree(node, level=0):
    indent = "  " * level
    area = node.box['width'] * node.box['height']
    print(f"{indent}Node {node.idx}: Area:{area}")
    for child in node.children:
        print_bounding_box_tree(child, level + 1)    

def find_node_by_idx(node, target_idx):
    if node.idx == target_idx:
        return node
    for child in node.children:
        result = find_node_by_idx(child, target_idx)
        if result:
            return result
    return None    

def count_nodes(roots):
    if not roots:
        return 0
    count = 0
    for node in roots:
        count += 1
        count += count_nodes(node.children)
    return count

def compute_heights(node):
    if not node.children:
        node.height = 0
        return 0
    else:
        heights = []
        for child in node.children:
            h = compute_heights(child)
            heights.append(h)
        node.height = 1 + max(heights)
        return node.height

def some_simple_checks(node):
    node = copy.deepcopy(node)
    new_children = []    
    for child in node.children:
        new_children.extend(some_simple_checks(child))

    if node.box['width'] >= 1920*0.8 or (node.box['height'] <=5 or node.box['width'] <=5):
        return new_children
    else:
        node.children = new_children
        return [node]

def simple_pruning(node):
    node = copy.deepcopy(node)    
    if node.height ==1:       
        children_count = len(node.children) 
        node.children = []
        node.height =0  
        return node,children_count
    else:
        return node,0

def edge_sharing_pruning(node):
    
    def has_common_edge(box1, box2):
        x1_left = box1['x']
        x1_right = box1['x'] + box1['width']
        y1_top = box1['y']
        y1_bottom = box1['y'] + box1['height']
        
        x2_left = box2['x']
        x2_right = box2['x'] + box2['width']
        y2_top = box2['y']
        y2_bottom = box2['y'] + box2['height']
        
        vertical_edge = (
            (x1_right == x2_left or x1_left == x2_right or x1_left == x2_left or x1_right == x2_right) and
            not (y1_bottom <= y2_top or y1_top >= y2_bottom)
        )
        
        horizontal_edge = (
            (y1_bottom == y2_top or y1_top == y2_bottom or y1_top == y2_top or y1_bottom == y2_bottom) and
            not (x1_right <= x2_left or x1_left >= x2_right)
        )
        return vertical_edge or horizontal_edge

    if not node.children:
        return node
    else:
        new_children = []   
        node = copy.deepcopy(node)   
        for i in range(len(node.children)):   
            if has_common_edge(node.children[i].box, node.box) and node.children[i].height <=1:
                pass
            else:
                new_child = edge_sharing_pruning(node.children[i])
                new_children.append(new_child)            
                
        node.children = new_children
        return node

def isSameBox(node1,node2):
    if node1.box['width']==node2.box['width'] and\
    node1.box['height']==node2.box['height']:
        return True
    else:
        return False


def are_trees_identical(node1, node2):
    if node1 is None and node2 is None:
        return True, {}
    
    if node1 is None or node2 is None:
        return False, {}
    
    if not isSameBox(node1, node2):
        return False, {}
    
    if len(node1.children) != len(node2.children):
        return False, {}
    
    matching = {node2.idx: node1.idx}
    used = [False] * len(node2.children)
    
    for child1 in node1.children:
        found_match = False
        for i, child2 in enumerate(node2.children):
            if not used[i]:
                marker, sub_matching = are_trees_identical(child1, child2)
                if marker:
                    used[i] = True
                    found_match = True
                    matching.update(sub_matching)
                    break
        
        if not found_match:
            return False, {}
    
    return True, matching   


# ---------------------------------------------------------------------------
# Route handler (unchanged)
# ---------------------------------------------------------------------------
async def route_handler(route):
    if route.request.resource_type == "font":
        await route.abort()
    else:
        await route.continue_()


# ---------------------------------------------------------------------------
# take_screenshot — NOW receives a context from the pool instead of creating
# its own.  The caller is responsible for acquire/release via the pool.
# ---------------------------------------------------------------------------
async def take_screenshot(point, index, scroll_length, path, url, context, url_dict, download_dict, attempt=0):       
    page = None
    popup_page = None
    #check if '-1' exists in url_dict
    if '-1' in url_dict.keys():
        return
    try:
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="load", timeout=20000)
            await page.wait_for_timeout(500)
        except Exception as e:
            print(f"Error in take_screenshot goto {url} number {Path(path).name}: {e}")
        
        try:
            await screenshot_by_scroll(page, '', save=False)            
        except Exception as e:
            print(f"Error in take_screenshot wait for images&screenshot scroll {url} number {Path(path).name}: {e}")
            return

        try:
            await page.wait_for_function(IMAGE_LOAD_CHECKER_ALL, timeout=15000)
            await page.screenshot(path=f"{path}/base_screenshots/base_{index}.jpg",type="jpeg", quality=50, scale="css", timeout=15000,full_page=True)
            # if not isSameScreenshot3(f"{path}/screenshot.jpg", await )[0]:
            if not isSameScreenshot2(open(f"{path}/screenshot.jpg", "rb").read(), open(f"{path}/base_screenshots/base_{index}.jpg", "rb").read())[0]:
                url_dict[index] = "this saves behaviour or cookies and base page changed after click, so skipping all clicks on this page"
                url_dict['-1'] = "this saves behaviour or cookies and base page changed after click, so skipping all clicks on this page"
                Path(f"{path}/base_screenshots/base_{index}.jpg").unlink(missing_ok=True)
                return
            #Below line is added after data is collected
            Path(f"{path}/base_screenshots/base_{index}.jpg").unlink(missing_ok=True)
        except Exception as e:
            print(f"Error in base page screenshot comparison number {Path(path).name} node {index}: {e}")
            # Don't return here - continue with the screenshot anyway

        download_happened = False
        last_download = None

        def on_download(download):
            nonlocal download_happened, last_download
            download_happened = True
            last_download = download

        page.on("download", on_download)

        # Listen for popups opened by THIS page only.
        # The "popup" event is page-scoped — it fires only when an action
        # on this exact page triggers a new tab/window.  Other tasks'
        # pages opening tabs will NOT fire this handler.
        def on_popup(new_page):
            nonlocal popup_page
            popup_page = new_page

        page.on("popup", on_popup)
        request_sent = False
        def handle_request(request):
            nonlocal request_sent
            request_sent = True

        page.on("request", handle_request)

        
        x, y = point
        await page.mouse.click(0, scroll_length)
        y -= scroll_length
        # if y > 1080:
        #     page_y = y
        #     await page.evaluate(
        #         f"(y) => window.scrollTo(0, {scroll_length})",
        #         page_y
        #     )
        #     viewport_y = await page.evaluate(
        #         "(y) => y - window.scrollY",
        #         page_y
        #     )
        #     y = viewport_y

        await page.evaluate("""
            window.domChanged = false;
            const observer = new MutationObserver(() => {
                window.domChanged = true;
            });
            observer.observe(document.body, { 
                attributes: true, 
                childList: true, 
                subtree: true 
            });
        """)
        page1_state = await get_page_state(page)
        
        await page.mouse.click(x, y)
        await page.wait_for_load_state("load", timeout=20000)
        await page.wait_for_function(IMAGE_LOAD_CHECKER_VIEWPORT, timeout=15000)
        await page.screenshot(
            path=f"{path}/screenshots/{index}.jpg",
            type="jpeg", quality=50, scale="css", timeout=15000,
        )
        page2_state = await get_page_state(page)    
        if popup_page or download_happened or request_sent or await page.evaluate("() => window.domChanged") or did_anything_change(page1_state, page2_state):
            redirected_url = page.url
        else:
            redirected_url = "nothing changed and this is empty space"

        if download_happened:
            if last_download.url not in download_dict:
                download_dict[last_download.url] = []
                Path(f"{path}/downloads").mkdir(parents=True, exist_ok=True)
                await last_download.save_as(f"{path}/downloads/{last_download.suggested_filename}")
            download_dict[last_download.url].append((index, last_download.suggested_filename))

        # If THIS page's click opened a new tab, screenshot that instead
        if popup_page:
            try:
                await popup_page.wait_for_load_state("load", timeout=20000)
                redirected_url = popup_page.url
                await popup_page.screenshot(
                    path=f"{path}/screenshots/{index}.jpg",
                    type="jpeg", quality=50, scale="css", timeout=15000,
                )
            except Exception:
                pass   # tab may have closed itself; keep the redirect URL

        url_dict[index] = redirected_url

    except PlaywrightError as e:
        if "context was destroyed" in str(e) or "pipe closed" in str(e).lower() or "target closed" in str(e).lower():
            print(f"Browser process issues for node {index} number {Path(path).name}: {e}")
            url_dict[index] = f"error: browser process closed/destroyed"
        else:
            print(f"Playwright error for node {index} number {Path(path).name}: {e}")
            url_dict[index] = f"error: {e}"
    except Exception as e:
        if "waiting for fonts to load..." in str(e):
            if attempt > 1:
                url_dict[index] = "error: too long to load fonts"
                print(f"Error taking screenshot for node {index} number {Path(path).name}: {e}")                
            else:
                # Close current page before retry to avoid accumulation
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass
                    page = None
                if popup_page:
                    try:
                        await popup_page.close()
                    except Exception:
                        pass
                    popup_page = None
                await take_screenshot(point, index, scroll_length, path, url, context, url_dict, download_dict, attempt + 1)
                return   # skip the finally-close below; the retry handled it
        else:
            print(f"Error taking screenshot for node {index} number {Path(path).name}: {e}")
            url_dict[index] = f"error: {e}"
    finally:
        # Close only the pages WE opened — never touch other tasks' pages
        if popup_page:
            try:
                await popup_page.close()
            except Exception:
                pass
        if page:
            try:
                await page.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Screenshot comparison & tree elimination (unchanged)
# ---------------------------------------------------------------------------
def isSameScreenshot(node1, node2, path):
    begin = time.time()
    img1 = np.array(Image.open(f"{path}/screenshots/{node1.idx}.jpg"))
    img2 = np.array(Image.open(f"{path}/screenshots/{node2.idx}.jpg"))    
    if not (img1.shape[0] == img2.shape[0] and img1.shape[1]== img2.shape[1]):
        return False, time.time() - begin
    marker = np.count_nonzero(img1 - img2) <= 25
    marker = marker or ssim(img1, img2) > 0.99999
    end = time.time()    
    return marker, end - begin

def isSameScreenshot2(bytes1, bytes2):
    begin = time.time()
    img1 = np.array(Image.open(BytesIO(bytes1)))
    img2 = np.array(Image.open(BytesIO(bytes2)))
    if not (img1.shape[0] == img2.shape[0] and img1.shape[1]== img2.shape[1]):
        return False, time.time() - begin
    marker = np.count_nonzero(img1 - img2) <= 25
    marker = marker or ssim(img1, img2) > 0.99
    end = time.time()    
    return marker, end - begin

def isSameScreenshot3(path1, bytes2):
    begin = time.time()
    img1 = np.array(Image.open(path1))
    img2 = np.array(Image.open(BytesIO(bytes2)))
    if not (img1.shape[0] == img2.shape[0] and img1.shape[1]== img2.shape[1]):
        return False, time.time() - begin
    marker = np.count_nonzero(img1 - img2) <= 25
    marker = marker or ssim(img1, img2) > 0.999
    end = time.time()    
    return marker, end - begin
    # return ssim(img1, img2), marker
    

def traverse_to_eliminate(node, reference_node, LargeMatching):
    if not node:
        return None
    
    node.visited = True
    node.keep = reference_node.keep
    new_children = []
    for child in node.children:        
        childreference_node = reference_node.node_storage[LargeMatching.get(child.idx)]
        child = traverse_to_eliminate(child, childreference_node, LargeMatching)        
        if child:
            new_children.append(child)
    node.children = new_children
    return node


def eliminate_redundant_nodes(root, path):
    if not root:
        return None

    actual_begin = time.time()        
    algo_time, img_time = 0., 0.
    
    for key in root.isomorphs:
        temp_node = root.node_storage[key]
        temp_node, tmp_alg_time, tmp_img_time = eliminate_redundant_nodes(temp_node, path)     
        if not Path(f"{path}/screenshots/{temp_node.idx}.jpg").exists():
            temp_node.keep = False
            marker = False
        else:
            marker, tmp_time = isSameScreenshot(temp_node, root, path)                       
            algo_time += tmp_alg_time
            img_time += tmp_img_time + tmp_time

        newnew_children = []                
        for child in temp_node.children:      
            if Path(f"{path}/screenshots/{temp_node.idx}.jpg").exists() and Path(f"{path}/screenshots/{child.idx}.jpg").exists():
                tmp_marker, tmp_time = isSameScreenshot(child, temp_node, path)
                img_time += tmp_time
            else:
                tmp_marker = False            
            
            child, tmp_alg_time, tmp_img_time = eliminate_redundant_nodes(child, path)
            algo_time += tmp_alg_time
            img_time += tmp_img_time
            if tmp_marker:
                child.keep = False
            newnew_children.append(child)
        temp_node.children = newnew_children

        if temp_node and marker:
            temp_node.keep = False

    newnew_children = []
    for k, indices in root.isomorphs.items():
        reference_node = root.node_storage[k]    
        for v in indices:
            if v == k:
                newnew_children.append(root.node_storage[k])
                continue
                    
            node_to_remove = find_node_by_idx(root, v)
            
            node_to_remove.keep = reference_node.keep
            if reference_node.keep:
                node_to_remove = traverse_to_eliminate(node_to_remove, reference_node, root.LargeMatching)
                newnew_children.append(node_to_remove)

    root.children = newnew_children
    actual_end = time.time()
    return root, actual_end - actual_begin - img_time, img_time

def root_level_isomorphism(roots, LargeMatching, node_storage, isomorphs, path):
    actual_begin = time.time()
    img_time = 0.
    D = {}
    for root in roots:
        if root.idx not in isomorphs.keys():
            D[root.idx] = root            
    
    for key in isomorphs:
        temp_node = node_storage[key]
        temp_node, tmp_alg_time, tmp_img_time = eliminate_redundant_nodes(temp_node, path)                                 
        img_time += tmp_img_time     

    new_children = []
    for k, indices in isomorphs.items():
        for v in indices:
            if v == k:
                new_children.append(node_storage[k])
                continue
            reference_node = node_storage[k]            
            node_to_remove = D[v]
            
            node_to_remove.keep = reference_node.keep
            if reference_node.keep:
                node_to_remove = traverse_to_eliminate(node_to_remove, reference_node, LargeMatching)
                new_children.append(node_to_remove)
    actual_end = time.time()
    return new_children, actual_end - actual_begin - img_time, img_time




async def collect_data(number, roots, url_dict, download_dict, url, path, context):

    scroll_info = json.load(open(f'{save_dir}/{number}/base_screenshots/metadata.json'))['scroll_steps']
    actual_begin = time.time()
    
    algo_time = 0.
    all_clicks = {}
    json_data = {}
    for root in roots:
        begin = time.time()
        json_data.update(root.get_keep_boxes())
        all_clicks.update(root.get_keep_clicks(scroll_info))
        end = time.time()
        algo_time += (end - begin)

    # Limit how many pages are open simultaneously inside the shared context
    # to avoid overwhelming the browser process.
    page_sem = asyncio.Semaphore(len(all_clicks) // 100 + 1)

    async def screenshot_task(point, idx, scroll_length):
        async with page_sem:
            try:
                # Add timeout to individual screenshot tasks
                await asyncio.wait_for(
                    take_screenshot(point, idx, scroll_length, path, url, context, url_dict, download_dict),
                    timeout=60  # 60 seconds per screenshot
                )
            except asyncio.TimeoutError:
                print(f"Screenshot timeout for node {idx} number {number}")
                url_dict[idx] = "error: screenshot timeout"
            except Exception as e:
                print(f"Screenshot task error for node {idx} number {number}: {e}")
                url_dict[idx] = f"error: {e}"

    tasks = []
    for idx, v in all_clicks.items():          
        point, scroll_length = v
        if Path(f"{path}/screenshots/{idx}.jpg").exists():
            continue
        tasks.append(screenshot_task(point, idx, scroll_length))

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await coro

    actual_end = time.time()

    for k in download_dict.keys():
        vt_val = isFileMalicious_existing(f'{path}/downloads/{download_dict[k][0][1]}')
        for (idx, filename) in download_dict[k]:
            json_data[idx]['isMalicious'] = vt_val
            json_data[idx]['MaliciousLabelSource'] = 'VirusTotal'
            json_data[idx]['isDownload'] = True
            json_data[idx]['downloadedFilename'] = download_dict[k][0][1]


    draw_boxes(f"{save_dir}/{number}/screenshot.jpg", roots, 'final', f"{save_dir}/{number}", draw_false_keep=False)
    for k in json_data.keys():
        json_data[k]['url'] = url_dict.get(k, "error: url not found in url_dict")

    json_data['base'] = url_dict['base']
    with open(f"{path}/data.json", "w") as f:
        json.dump(json_data, f)




async def screenshot_by_scroll(page, path, step=540, save=True):
    if save:
        Path(path).mkdir(parents=True, exist_ok=True)

    i = 0
    scroll_info = []
    prev_y = None

    MAX_SCROLL_STEPS = 15
    while i < MAX_SCROLL_STEPS:
        try:
            # 1. Ensure page is in a stable state before interacting
            await page.wait_for_load_state("domcontentloaded")

            # 2. Get current scroll and height data in one go to reduce context calls
            metrics = await page.evaluate("""
                () => {
                    return {
                        y: window.scrollY,
                        totalHeight: Math.max(
                            document.body?.scrollHeight || 0,
                            document.documentElement?.scrollHeight || 0
                        ),
                        viewport: window.innerHeight
                    }
                }
            """)
            
            y = metrics["y"]
            total_height = metrics["totalHeight"]
            viewport = metrics["viewport"]
            max_scroll = total_height - viewport

            # Logic to handle screenshots
            if save:
                shot_path = f"{path}/{i:04d}.jpg"
                await page.screenshot(
                    path=shot_path,
                    type="jpeg",
                    quality=50,
                    scale="css"
                )

            # Stop condition
            if y == prev_y or y >= max_scroll:
                break

            # Calculate and execute scroll
            actual_step = min(step, max_scroll - y)
            scroll_info.append(actual_step)

            await page.evaluate(f"window.scrollTo(0, {y + actual_step})")
            
            # 3. Add a small sleep to let lazy-loading/DOM shifts settle
            await page.wait_for_timeout(200) 

            prev_y = y
            i += 1

        except PlaywrightError as e:
            if "context was destroyed" in str(e):
                print(f"Navigation detected at step {i}. Attempting to recover...")
                await page.wait_for_load_state("networkidle")
                continue # Retry the loop with the new context
            else:
                raise e

    if save:
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump({"scroll_steps": scroll_info}, f)

    # Wrap the final scroll in a try-except as well
    try:
        await page.evaluate("window.scrollTo(0, 0)")
    except:
        pass

    



# ---------------------------------------------------------------------------
# single_link_collector — uses a SHARED context for the entire URL lifecycle.
# The caller (process_one_url) is responsible for acquire / release.
# ---------------------------------------------------------------------------
async def single_link_collector(number, url, context):    
    final_anns = []

    try:
        page = await context.new_page()
        Path(f"{save_dir}/{number}").mkdir(parents=True, exist_ok=True)

        try:
            await page.goto(url, wait_until="load", timeout=20000)
            await page.wait_for_timeout(500)        
        except Exception as e:
            print(f"Error in goto {url}: {e}")
            return
        try:
            await screenshot_by_scroll(page, f"{save_dir}/{number}/base_screenshots")
        except Exception as e:
            print("Error during initial scrolling/screenshot:", e)
        
        try:
            await page.wait_for_function(IMAGE_LOAD_CHECKER_ALL, timeout=15000)
        except Exception as e:
            print(f"Error waiting for images to load on {url} number {number}: {e}")
            return
        await page.screenshot(path=f"{save_dir}/{number}/screenshot.jpg", type="jpeg",
                    quality=50,
                    scale="css", timeout=15000, full_page=True)
        
        # with open(f'{save_dir}/{number}/base_screenshots/metadata.txt', 'w') as f:            
        #     f.write(f'{await page.evaluate("() => document.body.scrollHeight")}')

        
        await page.evaluate("""
                () => {
                try {
                    function isVisible(el) {
                    const style = getComputedStyle(el);
                    const r = el.getBoundingClientRect();
                    return style.display !== 'none' &&
                            style.visibility !== 'hidden' &&
                            style.opacity !== '0' &&
                            !el.hidden &&
                            el.getAttribute('aria-hidden') !== 'true' &&
                            r.width > 0 &&
                            r.height > 0;
                    }

                    function isNotOverlaid(el) {
                    const rect = el.getBoundingClientRect();
                    const centerX = rect.left + rect.width / 2;
                    const centerY = rect.top + rect.height / 2;
                    
                    const topElement = document.elementFromPoint(centerX, centerY);
                    
                    return topElement === el || el.contains(topElement);
                    }

                    function hasPointer(el) {
                    return getComputedStyle(el).pointerEvents !== 'none';
                    }

                    function isInteractable(el) {
                    const tag = el.tagName?.toLowerCase();
                    if (!tag) return false;

                    if (el.namespaceURI === "http://www.w3.org/2000/svg" && tag !== "svg") {
                        return false;
                    }

                    if (['a','button','select','textarea'].includes(tag)) return true;
                    if (tag === 'input' && el.type !== 'hidden') return true;
                    if (el.isContentEditable) return true;

                    const role = el.getAttribute('role');
                    if (['button','link','menuitem','option','tab','checkbox','radio','switch'].includes(role)) {
                        return true;
                    }

                    const style = getComputedStyle(el);
                    if (style.cursor === 'pointer') return true;

                    if (el.hasAttribute('onclick') || typeof el.onclick === 'function') return true;

                    return false;
                    }

                    let i = 0;
                    for (const el of document.querySelectorAll('*')) {
                    if (isVisible(el) && hasPointer(el) && isInteractable(el) ) {
                        el.setAttribute('data-interactable-idx', i++);
                    }
                    }
                } catch (e) {
                    console.error("Interactable tagging failed:", e);
                }
                }
                """)

        handles = await page.query_selector_all('[data-interactable-idx]')

        for h in handles:
            if not await h.is_visible():
                continue

            box = await h.bounding_box()
            if not box:
                continue

            data = await h.evaluate("""
            el => ({
                type: el.tagName.toLowerCase(),
                href:typeof el.href === "string" ? el.href : null,            
            })
            """)
            if box['x']>1920:
                continue
            ann = {
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
                "href": data["href"],
            }
            final_anns.append(ann)

        url_dict = {}
        download_dict = {}

        url_dict['base'] = page.url 
        # Done with the initial page — close the page but keep the context
        # alive for reuse in screenshot tasks and Stage 3.
        await page.close()
        
        scroll_info = json.load(open(f'{save_dir}/{number}/base_screenshots/metadata.json'))['scroll_steps']
        roots, TOTAL_NODES = build_bounding_box_tree(final_anns)

        draw_boxes(f"{save_dir}/{number}/screenshot.jpg", roots, 'before', f"{save_dir}/{number}")            
        
        begin = time.time()
        new_roots = []  
        simple_prune_count = 0
        for root in roots:
            new_root_set = some_simple_checks(root)
            for new_root in new_root_set:
                compute_heights(new_root)            
                new_root, sp_cnt = simple_pruning(new_root)
                simple_prune_count += sp_cnt
                new_root = edge_sharing_pruning(new_root)    
                new_roots.append(new_root)

        end = time.time()

        draw_boxes(f"{save_dir}/{number}/screenshot.jpg", new_roots, 'after', f"{save_dir}/{number}")
        roots = new_roots
        for root in roots:
            compute_heights(root)

        LargeMatching, node_storage, isomorphs = match(roots)

        total_efficient_click_time = 0.
        clicked_points = {}
        for k in isomorphs.keys():
            begin = time.time()
            efficient_clicks = node_storage[k].get_efficient_clicks(scroll_info)
            clicked_points.update(efficient_clicks)
            end = time.time()
            total_efficient_click_time += (end - begin)    

        path = f"{save_dir}/{number}"
        
        
        # ---- screenshot tasks sharing the same context ----
        # Limit concurrent pages within this context to avoid overwhelming
        # the browser process.
        page_sem = asyncio.Semaphore(len(clicked_points) // 100 + 1)

        async def screenshot_task(point, idx, scroll_length):
            async with page_sem:
                try:
                    # Add timeout to individual screenshot tasks in single_link_collector
                    await asyncio.wait_for(
                        take_screenshot(point, idx, scroll_length, path, url, context, url_dict, download_dict),
                        timeout=60  # 60 seconds per screenshot
                    )
                except asyncio.TimeoutError:
                    print(f"Screenshot timeout for node {idx} number {number}")
                    url_dict[idx] = "error: screenshot timeout"
                except Exception as e:
                    print(f"Screenshot task error for node {idx} number {number}: {e}")
                    url_dict[idx] = f"error: {e}"
        
        
        tasks = []
        for idx, (point, scroll_length) in clicked_points.items():                   
            if Path(f"{path}/screenshots/{idx}.jpg").exists():
                continue
            tasks.append(screenshot_task(point, idx, scroll_length))
        
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await coro
        
        return (number, new_roots, LargeMatching, node_storage, isomorphs, url_dict, download_dict)

    except Exception as e:
        print(f"Error processing URL {number} {url}: {e}")
        return number, e



# ---------------------------------------------------------------------------
# VirusTotal helpers (unchanged)
# ---------------------------------------------------------------------------
def sha256_of_file(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def isFileMalicious_existing(path):
    reqhash = sha256_of_file(path)
    url = f"https://www.virustotal.com/api/v3/files/{reqhash}"
    headers = {"accept": "application/json", "x-apikey": VT_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Error checking VirusTotal for {path}: {e}")
        return {}

async def get_page_state(page):
    return {
        "url": page.url,
        "screenshot": await page.screenshot(),
    }

def did_anything_change(page1, page2):
    if page1["url"] != page2["url"]:
        return True
    
    try:
        if not isSameScreenshot2(page1["screenshot"], page2["screenshot"])[0]:
            return True
    except:
        print("Error in did anything change screenshot comparison")

    return False

def clean_phishtank(phishtank):    
    D = {}
    for k,v in phishtank.items():
        if urlparse(k).netloc in ['l.ead.me','q-r.to','docs.google.com','qrco.de','linqapp.com','tinyurl.com','bit.ly','l.wl.co']:
            continue
        if 'Error:' in v:
            continue

        D[k] = v

    D2 = {}
    for k,v in D.items():
        D2[v] = k  

    D = {}
    for k,v in D2.items():
        D[k] = v

    return D

# ---------------------------------------------------------------------------
# JS image-load checkers (unchanged)
# ---------------------------------------------------------------------------
IMAGE_LOAD_CHECKER_ALL = r"""
            () => {
            const imgs = Array.from(document.images);

            return imgs
                .every(img => img.complete && img.naturalWidth > 0);
            }
            """

IMAGE_LOAD_CHECKER_VIEWPORT = r"""
            () => {
            const imgs = Array.from(document.images);

            const inViewport = (el) => {
                const r = el.getBoundingClientRect();
                return (
                r.bottom > 0 &&
                r.right > 0 &&
                r.top < window.innerHeight &&
                r.left < window.innerWidth
                );
            };

            return imgs
                .filter(inViewport)
                .every(img => img.complete && img.naturalWidth > 0);
            }
            """


# ---------------------------------------------------------------------------
# Utility / debug helpers (unchanged)
# ---------------------------------------------------------------------------
def print_keeps(roots):
    def traverse(node, level=0):
        indent = "  " * level
        print(f"{indent}Node {node.idx}: Keep={node.keep}")
        for child in node.children:
            traverse(child, level + 1)
    for r in roots:
        traverse(r)

def save_nodes(roots, filename):
    all_nodes = []
    def traverse(node):
        all_nodes.append({
            "idx": node.idx,
            "box": node.box
        })
        for child in node.children:
            traverse(child)
    for r in roots:
        traverse(r)
    import json
    with open(filename, "w") as f:
        json.dump(all_nodes, f, indent=4)

def has_empty_space(large_box, small_boxes):
    """
    large_box: (x, y, w, h)
    small_boxes: [(x, y, w, h), ...]
    Returns a point (x, y) in large_box not covered by small_boxes if it exists,
    otherwise returns None.
    """
    if len(small_boxes) == 0:
        L_x1, L_y1, L_w, L_h = large_box
        return (L_x1 + L_w / 2, L_y1 + L_h / 2)

    L_x1, L_y1, L_w, L_h = large_box
    L_x2, L_y2 = L_x1 + L_w, L_y1 + L_h

    x_coords = {L_x1, L_x2}
    y_coords = {L_y1, L_y2}

    for (x, y, w, h) in small_boxes:
        if L_x1 < x < L_x2: x_coords.add(x)
        if L_x1 < x + w < L_x2: x_coords.add(x + w)
        if L_y1 < y < L_y2: y_coords.add(y)
        if L_y1 < y + h < L_y2: y_coords.add(y + h)

    sorted_x = sorted(list(x_coords))
    sorted_y = sorted(list(y_coords))

    for i in range(len(sorted_x) - 1):
        for j in range(len(sorted_y) - 1):
            mid_x = (sorted_x[i] + sorted_x[i+1]) / 2
            mid_y = (sorted_y[j] + sorted_y[j+1]) / 2

            is_covered = False
            for (sx, sy, sw, sh) in small_boxes:
                if sx <= mid_x <= sx + sw and sy <= mid_y <= sy + sh:
                    is_covered = True
                    break
            
            if not is_covered:
                return (mid_x, mid_y)

    return None
        




############################################
############ URL deduplication utils #######
############################################

#Currently, these are placeholders.
def get_index(url):
    return random.randint(0, 1000000)

def get_url(index):
    #random string of length 10
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))

def deduplicate_urls(urls,base_folder):
    seen = set()
    unique_urls = []
    for url in urls:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if domain not in seen:
            seen.add(domain)
            unique_urls.append(url)

    image_similarities = defaultdict(list)
    for url in unique_urls:
        index = get_index(url)
        filename = f"{base_folder}/{index}/screenshot.jpg"

        similar = False 
        image1 = np.array(Image.open(filename))
        for key in image_similarities.keys():
            image2 = np.array(Image.open(f"{base_folder}/{key}/screenshot.jpg"))
            similarity = ssim(image1, image2, full=True)[0]
            if similarity > 0.99999:
                image_similarities[key].append(url)
                similar = True
                break

        if not similar:
            image_similarities[index].append(url)

    return map(lambda x: get_url(x), image_similarities.keys())