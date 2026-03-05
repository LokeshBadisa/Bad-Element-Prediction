############################################
############ Data preprocessing utils ######
############################################
import re
import io
import base64
import numpy as np
import cv2
import json
from scipy.stats import entropy
import colorsys
from PIL import Image, ImageDraw
from collections import defaultdict
from pathlib import Path
import matplotlib.colors as mplc
import matplotlib as mpl
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def isInImage(y,height, img_num, scroll_info):
    img_top = sum(scroll_info[:img_num])
    img_bottom = img_top + 1080
    box_top = y
    box_bottom = y + height
    return not (box_bottom < img_top or box_top > img_bottom)

def extract_answer(text):
    match = re.findall(r'<answer>(.*?)</answer>', text)
    # print(match)
    return match



def estimate_text_bbox(text, font_size, x, y, pad_factor=0.7, scale=1.0):
    #xdifference = 10.5
    #ydifference = 18
    x0,y0,x1,y1 = x,y-18.0,x+10.5,y
    return x0,y0,x1,y1

def find_intersection_area(t1,t2):
    x1 = max(t1[0], t2[0])
    y1 = max(t1[1], t2[1])
    x2 = min(t1[2], t2[2])
    y2 = min(t1[3], t2[3])
    
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0.0

def numpy_to_base64(rgb_array):
    img = Image.fromarray(rgb_array)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)

    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64

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

def has_text_or_icon(image, x1, y1, x2, y2):
    roi = image[x1:x2, y1:y2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # 2. Intensity variance
    variance = np.var(gray)

    # Heuristic decision
    if edge_density > 0.02 and variance > 15:
        return True
    else:
        return False

def get_text_color(rgb_tuple):
    r, g, b = rgb_tuple[:3] # Ensure we only take RGB even if Alpha is present
    
    # Perceptual Luminance Formula
    # Using the standard ITU-R BT.601 coefficients
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    
    # If the background is bright (luma > 0.5), use black text. 
    # Otherwise, use white.
    return "black" if luma > 0.5 else "white"


def get_max_contrast_color(color1, color2):
    """
    Inputs: color1, color2 as normalized RGB tuples (0.0 - 1.0)
    Output: A third color tuple (r, g, b)
    """
    # 1. Find the midpoint (average)
    mid_r = (color1[0] + color2[0]) / 2
    mid_g = (color1[1] + color2[1]) / 2
    mid_b = (color1[2] + color2[2]) / 2
    
    # 2. Invert the midpoint to get the "opposite" color
    # This ensures the new color is as far as possible from the average
    inv_r = 1.0 - mid_r
    inv_g = 1.0 - mid_g
    inv_b = 1.0 - mid_b
    
    return (inv_r, inv_g, inv_b)

def get_vibrant_separator(color1, color2):
    # Get the inverted midpoint as before
    r, g, b = get_max_contrast_color(color1, color2)
    
    # Convert to HSV to boost visibility
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Max out Saturation and Value so the line is "Neon"
    return colorsys.hsv_to_rgb(h, 1.0, 1.0)



def get_info_loss_score(image_rgb, box):    
    # 1. Crop and Grayscale
    x1, y1, x2, y2 = map(int, box)    
    crop = image_rgb[y1:y2, x1:x2]
    var_r = np.var(crop[:,:,0])
    var_g = np.var(crop[:,:,1])
    var_b = np.var(crop[:,:,2])
    
    # Use the maximum variance across channels
    max_var = max(var_r, var_g, var_b)
    
    if max_var < 15.0:
        return 0.0
    
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    
    # 2. Noise Reduction (Crucial for "Plain" backgrounds)
    # This removes compression artifacts that look like "info"
    gray_smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    

    # 4. Entropy (Information Theory)
    # Normalized to 0-1 (8 bits max entropy is 8.0)
    hist = cv2.calcHist([gray_smooth], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    s_entropy = entropy(hist_norm, base=2) / 8.0
    
    # 5. Edge Density (Structural Information)
    # Canny detects the outlines of icons and text characters
    edges = cv2.Canny(gray_smooth, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 6. Combined Weighted Score
    # We weigh Edge Density higher because icons/text have high edge-to-area ratios
    raw_score = (s_entropy * 0.4) + (edge_density * 0.6)
    
    # Scale to 0-100 and Clip
    final_score = np.clip(raw_score * 150, 0, 100)
    
    return round(float(final_score), 2)


class SoM():
    def __init__(self, base_screenshots_folder, data_json_path, scroll_info_json_path):
        self.base_screenshots_folder = base_screenshots_folder  
        self.imgs_paths = sorted(list(Path(base_screenshots_folder).glob("*.jpg")), key=lambda x: int(x.stem))      
        self.data = json.load(open(data_json_path))
        self.data = {k:v for k,v in self.data.items() if k!='base' and 'error' not in v['url'].lower() and 'nothing changed and this is empty space' not in v['url'].lower()}
        self.outputs = [VisImage(np.asarray(Image.open(img_path)).clip(0, 255).astype(np.uint8), scale=1.0) for img_path in self.imgs_paths]
        self._default_font_size = 12
        self.scroll_info = json.load(open(scroll_info_json_path))['scroll_steps']
        self.boxes_in_image = defaultdict(list) #this and boxes_in_image of process() are different. 
        # This one is used to bbox of element whereas process()'s is used to store textboxes to avoid intersection when placing textboxes.          
        
        self.inimagedict = {}
        for img_num, img_path in enumerate(self.imgs_paths):
            for k,v in self.data.items():
                area = find_intersection_area(
                        (v["x"], v["y"]  - sum(self.scroll_info[:img_num]), v["x"] + v["width"], v["y"] + v["height"] - sum(self.scroll_info[:img_num])),
                        (0, 0, self.outputs[img_num].width, self.outputs[img_num].height)
                    )
                self.inimagedict[(k, img_num)] = isInImage(v["y"], v["height"], img_num, self.scroll_info)\
                and area >= 0.8*(v["width"]*v["height"])
                

        for k,v in self.inimagedict.items():
            if v:
                self.boxes_in_image[k[1]].append(k[0])

        self.process()

    def least_intersection_position(self,x,y,text,img_num,options):
        
        D = defaultdict(float)
        for i,option in enumerate(options):
            x1,y1,x2,y2 = estimate_text_bbox(text, self._default_font_size, option[0], option[1])
            if x1 < 0 or y1 - sum(self.scroll_info[:img_num]) < 0 or x2 > self.outputs[img_num].width or y2 - sum(self.scroll_info[:img_num]) > self.outputs[img_num].height:
                continue
            running_sum = 0.0
            for box in self.boxes_in_image[img_num]:
                if box==text:
                    continue
                bx1,by1,bx2,by2 = self.data[box]["x"], self.data[box]["y"]- sum(self.scroll_info[:img_num]), self.data[box]["x"] + self.data[box]["width"], self.data[box]["y"] + self.data[box]["height"] - sum(self.scroll_info[:img_num])
                
                int_area = find_intersection_area(
                    (x1, y1 - sum(self.scroll_info[:img_num]), x2, y2 - sum(self.scroll_info[:img_num])),
                    (bx1, by1, bx2, by2)
                )
                running_sum += int_area
            D[i] = running_sum

        
        #Check if all values in D are equal
        all_equal = True if len(set(list(D.values()))) == 1 else False
        if all_equal:
            return options[0]
        
        best_option = options[sorted(D.items(), key=lambda item: item[1])[0][0]]
        return best_option

    def get_options(self, x, y, boxwidth, text, boxes, img_num):
        # This function removes options that intersect with existing boxes
        
        width = len(text) * 10.5
        #Options
        options = [
            (x, y - 5),  # Top left
            # (x + width -5 , y - 5),  # Top right
            # (x + width +5,y),##right top
            # (x + width +5,y + height - 5),  # Bottom right
            (x-width,y+15), #Left top
            (x+boxwidth-width,y-5), #Top right
            (x+boxwidth+5,y+20) #Right top
        ]
        final_options = []
        
        for option in options:
            x1,y1,x2,y2 = estimate_text_bbox(text, self._default_font_size, option[0], option[1])
            if x1<0 or x1>1920 or y1-sum(self.scroll_info[:img_num])<0 or y1-sum(self.scroll_info[:img_num])>1080:
                continue
            if x2<0 or x2>1920 or y2-sum(self.scroll_info[:img_num])<0 or y2-sum(self.scroll_info[:img_num])>1080:
                continue
            mark = True
            for box in boxes:
                bx1,by1,bx2,by2 = box
                int_area = find_intersection_area(
                    (x1, y1, x2, y2),
                    (bx1, by1, bx2, by2)
                )
                if int_area > 0:
                    mark = False
                    break
            if mark:
                final_options.append(option)
        
        
        final_options = sorted(final_options, key=lambda opt: get_info_loss_score(self.outputs[img_num].img, estimate_text_bbox(text, self._default_font_size, opt[0], opt[1]-sum(self.scroll_info[:img_num]))))
        
        return final_options

    def process(self):
        #segment each image based on the coordinates and save the segmented images in self.seg_imgs
        alpha = 0.1

        for i,img_path in enumerate(self.imgs_paths):      
            boxes_in_image = []
            for k,v in sorted(self.data.items(), key=lambda item: item[1]['width']*item[1]['height']):
                if self.inimagedict[(k, i)]:    
                    
                    x1,y1,x2 = v["x"], v["y"] - sum(self.scroll_info[:i]),v["x"] + v["width"]                    
                    outmidpoint = ((x1+x2)/2, y1-2)
                    inmidpoint = ((x1+x2)/2, y1+2)
                    color1 = self.outputs[i].img[int(inmidpoint[1]), int(inmidpoint[0])]/255.0
                    color2 = self.outputs[i].img[int(outmidpoint[1]), int(outmidpoint[0])]/255.0
                    color = get_vibrant_separator(color1, color2)
                    
                    options = self.get_options(v["x"], v["y"], v["width"], k, boxes_in_image, i)     
                    if len(options) == 0:
                        continue             

                    self.outputs[i].ax.add_patch(mpl.patches.Rectangle(
                        (v["x"], v["y"] - sum(self.scroll_info[:i])),
                        v['width'],
                        v['height'],
                        fill=False,
                        edgecolor=mplc.to_rgb(color) + (alpha,),
                        facecolor='none',
                        linewidth=2,
                        linestyle='dashed',
                        alpha=1.0
                    ))
                    
                    #write the number k in the top left of the box outside the box. text color should be white and background color should be the same as the box color but with alpha 0.8
                    posn = self.least_intersection_position(v['x'], v['y'], k, i, options)
                    
                    self.outputs[i].ax.text(
                        posn[0],
                        posn[1]-sum(self.scroll_info[:i]),
                        k,
                        fontsize=self._default_font_size * self.outputs[i].scale,
                        family="sans-serif",
                        bbox={"facecolor": mplc.to_rgb(color) + (0.8,), "alpha": 1.0, "pad": 0.7, "edgecolor": "none"},                        
                        color=get_text_color(color),                        
                        ha="left",
                        va="bottom" 
                    )
                                            
                    boxes_in_image.append(estimate_text_bbox(k, self._default_font_size, posn[0], posn[1]))
                    
    
    def save(self, output_folder):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for i, output in enumerate(self.outputs):
            if len(self.boxes_in_image[i])>0:                
                print(f"Saving image {i} to {output_folder}/{i}.jpg")
                output.save(f"{output_folder}/{i}.jpg")

def get_img_num(y,height, scroll_info):    
    centers = [540+sum(scroll_info[:i]) for i in range(len(scroll_info)+1)]
    box_center = y + height / 2
    min_dist = float('inf')
    img_num = -1
    for i, center in enumerate(centers):
        dist = abs(center - box_center)
        if dist < min_dist:
            min_dist = dist
            img_num = i
    return img_num

def highlight_box_on_image(img_path, x,y,width,height):    
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + width, y + height], outline="red", width=2)
    img.save("tmp_img.jpg")

