import json
from tqdm import tqdm
import copy
import matplotlib.colors as mplc
import matplotlib as mpl
from pathlib import Path
from PIL import Image
from collections import defaultdict
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# def get_img_num(y,height, num_imgs):    
#     centers = [540*i + 540 for i in range(num_imgs)]
#     box_center = y + height / 2
#     min_dist = float('inf')
#     img_num = -1
#     for i, center in enumerate(centers):
#         dist = abs(center - box_center)
#         if dist < min_dist:
#             min_dist = dist
#             img_num = i
    # return img_num

def isInImage(y,height, img_num):
    img_top = 540*img_num
    img_bottom = img_top + 1080
    box_top = y
    box_bottom = y + height
    return not (box_bottom < img_top or box_top > img_bottom)

class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")    


def estimate_text_bbox(text, font_size, x, y, scale=1.0):
    avg_char_width = font_size * scale  # realistic
    width = len(text) * avg_char_width
    height = font_size * scale

    # because va="bottom", ha="left"
    x0 = x
    y0 = y - height
    x1 = x + width
    y1 = y #+ height

    return x0, y0, x1, y1

def find_intersection_area(t1,t2):
    """
    Find the intersection area of two rectangles.
    Args:
        t1 (tuple): (x1, y1, x2, y2) coordinates of the first rectangle
        t2 (tuple): (x1, y1, x2, y2) coordinates of the second rectangle
    Returns:
        float: area of intersection
    """
    x1 = max(t1[0], t2[0])
    y1 = max(t1[1], t2[1])
    x2 = min(t1[2], t2[2])
    y2 = min(t1[3], t2[3])
    
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0.0

class SoM():
    def __init__(self, base_screenshots_folder, data_json_path, scroll_info_json_path, print_text=False):
        self.base_screenshots_folder = base_screenshots_folder  
        self.imgs_paths = sorted(list(Path(base_screenshots_folder).glob("*.jpg")))      
        self.data = json.load(open(data_json_path))
        self.data = {k:v for k,v in self.data.items() if k!='base'}
        self.outputs = [VisImage(np.asarray(Image.open(img_path)).clip(0, 255).astype(np.uint8), scale=1.0) for img_path in self.imgs_paths]
        self._default_font_size = 12
        self.scroll_info = json.load(open(scroll_info_json_path))['scroll_steps']
        
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        self.color_proposals = [list(mcolors.to_rgb(c)) for c in colors]

        self.inimagedict = {}
        for img_num, img_path in enumerate(self.imgs_paths):
            for k,v in self.data.items():
                self.inimagedict[(k, img_num)] = isInImage(v["y"], v["height"], img_num)

        self.process(print_text)

    def least_intersection_position(self,x,y,width,height,text,img_num,print_text=False):
        avg_char_width = self._default_font_size
        width = len(text) * avg_char_width
        #Options
        options = [
            (x, y - 5),  # Top left
            # (x + width -5 , y - 5),  # Top right
            # (x + width +5,y),##right top
            # (x + width +5,y + height - 5),  # Bottom right
            (x-width,y+15)
        ]
        D = defaultdict(list)
        for i,option in enumerate(options):
            x1,y1,x2,y2 = estimate_text_bbox(text, self._default_font_size, option[0], option[1])
            for k,v in self.data.items():
                if self.inimagedict[(k, img_num)] and (k!=text):
                    int_area = find_intersection_area(
                        (x1, y1 - img_num * 540, x2, y2 - img_num * 540),
                        (v["x"], v["y"] - img_num * 540, v["x"] + v["width"], v["y"] + v["height"] - img_num * 540)
                    )
                    if int_area > 0:
                        D[i].append(int_area)

        for i in range(len(options)):
            if i not in D:
                D[i] = [0.0]  # Ensure every option has an entry in D, even if empty
        if print_text:
            print(img_num, text, D)
        # Find the option with least average intersection area
        min_avg = float('inf')
        best_option = None
        for i, areas in D.items():
            avg_area = np.mean(areas)
            if avg_area < min_avg:
                min_avg = avg_area
                best_option = options[i]
        # print(f"Best option for text '{text}' at image {img_num}: {D}")
        if best_option is None:
            # If no best option found, return the first option
            best_option = options[0]
        return best_option

    def process(self, print_text=False):
        #segment each image based on the coordinates and save the segmented images in self.seg_imgs
        alpha = 0.1

        for i,img_path in enumerate(self.imgs_paths):            
            for k,v in self.data.items():
                if self.inimagedict[(k, i)]:
                    color = self.color_proposals[np.random.randint(0, len(self.color_proposals))]                    
                    self.outputs[i].ax.add_patch(mpl.patches.Rectangle(
                        (v["x"], v["y"] - sum(self.scroll_info[:i])),
                        v["width"],
                        v["height"],
                        fill=False,
                        edgecolor=mplc.to_rgb(color) + (alpha,),
                        facecolor='none',
                        linewidth=2,
                        linestyle='dashed',
                        alpha=1.0
                    ))
                    #write the number k in the top left of the box outside the box. text color should be white and background color should be the same as the box color but with alpha 0.8
                    posn = self.least_intersection_position(v['x'], v['y'], v['width'], v['height'], k, i, print_text)
                    self.outputs[i].ax.text(
                        posn[0],
                        posn[1]-sum(self.scroll_info[:i]),
                        k,
                        size=self._default_font_size * self.outputs[i].scale,
                        family="sans-serif",
                        bbox={"facecolor": mplc.to_rgb(color) + (0.8,), "alpha": 1.0, "pad": 0.7, "edgecolor": "none"},                        
                        color='white',
                        zorder=10,
                        ha="left",
                        va="bottom" 
                    )
    
    def save(self, output_folder):
        for i, output in enumerate(self.outputs):
            output.save(f"{output_folder}/{i}.jpg")

# sommer = SoM('/data1/lokesh/blp/appropriate/32/32/base_screenshots','/data1/lokesh/blp/appropriate/32/32/data.json')
# sommer = SoM('/data1/lokesh/blp/32/base_screenshots','/data1/lokesh/blp/32/data.json','/data1/lokesh/blp/32/base_screenshots/metadata.json')    
# sommer.process()


SYSTEM_PROMPT = """
                    You are an assistant that classifies the intent of each bounding box in a screenshot of a conversation.

                    For every bounding box, analyze the visible content and infer its intent using contextual cues such as semantics, sentiment, and conversational role. Use only the information present in the image and do not assume any external context.

                    Assign exactly one label to each bounding box from the following set:
                    - malicious: Content that involves harmful intent (e.g., scams, malware, harassment, fraud, explicit wrongdoing).
                    - benign: Normal, safe, harmless, or informational content.
                    - unknown: Insufficient information, unreadable content, or ambiguous intent.

                    Output format (strict):
                    Return a single valid JSON object of the form:
                    { box_number: label, ... }
                    Box numbers should be in ascending order

                    Rules:
                    - All bounding boxes must be included in the same JSON object.
                    - Each bounding box must have exactly one label.
                    - Do not produce multiple JSON objects.
                    - Do not include explanations, markdown, or any extra text in the output.
                    - If the intent cannot be confidently determined, label it as unknown.
                    """


sharegpt = []
for folder in tqdm(sorted(Path('combineddata').iterdir())[:-100]):
        # if folder.name not in ['1904']:
        #     continue
    # if Path(folder/'final_boxes.jpg').exists():        
        jsondata = json.load(open(folder/f'data.json'))
        if 'base' not in jsondata:
            print(f"Skipping {folder} as 'base' key is missing in data.json")
            continue
        labelsdata = json.load(open(folder/f'answers.json'))
        scrolldata = json.load(open(folder/f'base_screenshots/metadata.json'))['scroll_steps']

        temp_dict_base = {"messages": [],"images": []}
        temp_dict_base["messages"].append(
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        )
        Path(f'traindata_wurl/{folder.name}').mkdir(parents=True, exist_ok=True)
        sommer = SoM(folder/f'base_screenshots',folder/f'data.json',folder/f'base_screenshots/metadata.json')#,True if folder.name in ['1904'] else False)    
        sommer.save(f'traindata_wurl/{folder.name}')
        for i in range(len(scrolldata)+1):
            boxes_in_image = [int(key) for key in labelsdata.keys() if sommer.inimagedict[(key, i)]]
            if len(boxes_in_image) == 0:    
                continue
            # for key in sorted(boxes_in_image):                 
            temp_dict = copy.deepcopy(temp_dict_base)
            # temp_dict["messages"].append(
            #     {
            #         "role": "user",
            #         "content": "<image>"
            #     }
            # )
            temp_dict["messages"].append(
                {
                    "role": "human",
                    "content": f"<image> Classify the given image. URL of this page is {jsondata['base']}"
                }
            )
            temp_dict["messages"].append(
                {
                    "role": "gpt",
                    "content": '{'+', '.join(f'{key}:{labelsdata[str(key)][-1]}' for key in sorted(boxes_in_image))+'}'
                }
            )
            temp_dict["images"].append(f'/data1/lokesh/bep/data_gen/traindata_wurl/{folder.name}/{i}.jpg')
            sharegpt.append(temp_dict)

with open("traindata_wurl/sharegpt_data.json", "w") as f:
    json.dump(sharegpt, f, indent=4)    