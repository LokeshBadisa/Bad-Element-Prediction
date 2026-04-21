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
from fast_ssim import ssim
from PIL import Image, ImageDraw
from collections import defaultdict
from pathlib import Path
import matplotlib.colors as mplc
import matplotlib as mpl
import matplotlib.figure as mplfigure
from color_descriptor import describe_color
from matplotlib.backends.backend_agg import FigureCanvasAgg


def isInImage(y,height, img_num, prefix_sum):
    img_top = prefix_sum[img_num]
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

def compute_ssim(img1, img2, box1, box2, ssim_dict, key, save=False):
    # load images as RGB
    # img1 = Image.open(img1_path)
    # img2 = Image.open(img2_path)
    
    # crop using PIL (x1, y1, x2, y2)
    # crop1 = img1.crop(box1)
    # crop2 = img2.crop(box2)
    #crop if img1 and img2 are np.arrays
    #typecast box coordinates to int
    box1 = tuple(map(int, box1))
    box2 = tuple(map(int, box2))
    arr1 = img1[box1[1]:box1[3], box1[0]:box1[2]]
    arr2 = img2[box2[1]:box2[3], box2[0]:box2[2]]

    # ensure same size
    if arr1.shape[:2] != arr2.shape[:2]:
        ssim_dict[key] = 0.0
        return 0.0

    # convert to numpy arrays
    # arr1 = np.array(crop1)
    # arr2 = np.array(crop2)
    # if save:
    #     #save both crops for debugging
    #     Image.fromarray(arr1).save('crop1.jpg')
    #     Image.fromarray(arr2).save('crop2.jpg')

    # compute color SSIM
    score = ssim(arr1, arr2)
    ssim_dict[key] = score

    return score

def image_ssim(img1, img2):
    img1 = Image.open(img1)
    img2 = Image.open(img2)

    if img1.size != img2.size:
        return 0.0
    
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return ssim(arr1, arr2)

class SoM():
    def __init__(self, base_screenshots_folder, data_json_path: Path, scroll_info_json_path, process_all_boxes=False, process_each_box=False, process_eb_folder=None, crop_boxes=False, crop_location=None, print_box=None):
        self.base_screenshots_folder = base_screenshots_folder  
        self.imgs_paths = sorted(list(Path(base_screenshots_folder).glob("*.jpg")), key=lambda x: int(x.stem))      
        self.data = json.load(open(data_json_path))
        self.base_url = self.data['base']
        self.data = {k:v for k,v in self.data.items() if k!='base' and 'error' not in v['url'].lower() and 'nothing changed and this is empty space' not in v['url'].lower()}
        self.outputs = [VisImage(np.asarray(Image.open(img_path)).clip(0, 255).astype(np.uint8), scale=1.0) for img_path in self.imgs_paths]
        self._default_font_size = 12
        self.scroll_info = json.load(open(scroll_info_json_path))['scroll_steps']        
        self.boxes_in_image = defaultdict(list) #this and boxes_in_image of process() are different. 
        # This one is used to bbox of element whereas process()'s is used to store textboxes to avoid intersection when placing textboxes.          
        base_img_path = f'{data_json_path.parent}/screenshot.jpg'
        base_img = np.asarray(Image.open(base_img_path))
        img_cache = {img_num: np.asarray(Image.open(img_path)) for img_num, img_path in enumerate(self.imgs_paths)}
        self.prefix_sum = prefix_sum = [sum(self.scroll_info[:img_num]) for img_num in range(len(self.imgs_paths))]
        self.ssim_dict = {}

        #default value should be false for all keys
        self.inimagedict = defaultdict(lambda: False)
        for img_num, img_path in enumerate(self.imgs_paths):
            for k,v in self.data.items():
                if isInImage(v["y"], v["height"], img_num, prefix_sum):
                    area = find_intersection_area(
                            (v["x"], v["y"]  - prefix_sum[img_num], v["x"] + v["width"], v["y"] + v["height"] - prefix_sum[img_num]),
                            (0, 0, self.outputs[img_num].width, self.outputs[img_num].height)
                        )
                    
                    base_img_coords = [v['x'], v['y'], v['x'] + v['width'], v['y'] + v['height']]
                    curr_img_coords = [v['x'], v['y'] - prefix_sum[img_num], v['x'] + v['width'], v['y'] + v['height'] - prefix_sum[img_num]]
                    # base_img_path = f'{data_json_path.parent}/screenshot.jpg'
                    # result_img_path = f'{data_json_path.parent}/screenshots/{img_num}.jpg'
                    # val1 = isInImage(v["y"], v["height"], img_num, self.scroll_info)
                    # val2 = area >= 0.8*(v["width"]*v["height"])
                    # val3 = compute_ssim(base_img, img_cache[img_num], base_img_coords, curr_img_coords,self.ssim_dict, f'{img_num}_{k}') 
                    # val4 = val3 > 0.85                
                    # if print_box is not None and k == print_box:
                    #     print(f'Image {img_num}: {val1}, {val2}, {val3}, {val4}')

                    #Reason behind the following:
                    #1. Selected box should be in current image
                    #2. Area of the box in the current image should be at least 80% of the total area of the box (to filter out boxes that are mostly out of the image)
                    #3. The box in the full image and the box in the current image should be almost the same                
                    self.inimagedict[(k, img_num)] = area >= 0.8*(v["width"]*v["height"]) and\
                         compute_ssim(base_img, img_cache[img_num], base_img_coords, curr_img_coords, self.ssim_dict, f'{img_num}_{k}') > 0.85
                    # self.inimagedict[(k, img_num)] = val1 and val2 and val4
                    # and image_ssim(img_path, result_img_path) < 0.95
                    

        for k,v in self.inimagedict.items():
            if v:
                self.boxes_in_image[k[1]].append(k[0])

        if process_all_boxes:
            self.process_all_boxes()
        
        if process_each_box:
            self.color_list = {}
            self.process_eb_folder = process_eb_folder
            Path(self.process_eb_folder).mkdir(parents=True, exist_ok=True)
            self.process_each_box()

        if crop_boxes:
            Path(crop_location).mkdir(parents=True, exist_ok=True)
            self.crop_boxes(crop_location)

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

    def process_all_boxes(self):
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
                    
    def process_each_box(self):
        alpha = 0.1
        
        for i,img_path in enumerate(self.imgs_paths):      
            boxes_in_image = []
            for k,v in sorted(self.data.items(), key=lambda item: item[1]['width']*item[1]['height']):
                if self.inimagedict[(k, i)]:  
                    x1,y1,x2 = v["x"], v["y"] - sum(self.scroll_info[:i]),v["x"] + v["width"]                    
                    outmidpoint = ((x1+x2)/2, y1-2)
                    inmidpoint = ((x1+x2)/2, y1+2)
                    output = VisImage(np.asarray(Image.open(img_path)).clip(0, 255).astype(np.uint8), scale=1.0)
                    color1 = output.img[int(inmidpoint[1]), int(inmidpoint[0])]/255.0
                    color2 = output.img[int(outmidpoint[1]), int(outmidpoint[0])]/255.0
                    color = get_vibrant_separator(color1, color2)
                    self.color_list[f'{i}_{k}'] = describe_color(color)
                    
                    options = self.get_options(v["x"], v["y"], v["width"], k, boxes_in_image, i)     
                    if len(options) == 0:
                        continue  
                    
                    output.ax.add_patch(mpl.patches.Rectangle(
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
                    
                    output.ax.text(
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

                    Path(f'{self.process_eb_folder}/{i}').mkdir(parents=True, exist_ok=True)
                    output.save(f'{self.process_eb_folder}/{i}/{k}.jpg')

    def crop_boxes(self, crop_location):
        crop_marker = defaultdict(lambda: 0.0)   
        ssim_marker = defaultdict(lambda: 0.0)        
        for img_num, img_path in enumerate(self.imgs_paths):
            img = Image.open(img_path)
            for k, v in self.data.items():
                if isInImage(v["y"], v["height"], img_num, self.prefix_sum):
                # if self.inimagedict[(k, img_num)]:
                    scroll_offset = self.prefix_sum[img_num]
                    x1 = v["x"]
                    y1 = v["y"] - scroll_offset
                    x2 = v["x"] + v["width"]
                    y2 = v["y"] + v["height"] - scroll_offset

                    x1 = max(0, min(x1, self.outputs[img_num].width))
                    y1 = max(0, min(y1, self.outputs[img_num].height))
                    x2 = max(0, min(x2, self.outputs[img_num].width))
                    y2 = max(0, min(y2, self.outputs[img_num].height))
                    area = (x2 - x1) * (y2 - y1)
                    if area > crop_marker[k] and self.ssim_dict.get(f'{img_num}_{k}', 0.0) > 0.7:                        
                        crop = img.crop((x1, y1, x2, y2))
                        crop.save(f'{crop_location}/{k}.jpg')
                        crop_marker[k] = area

    def save(self, output_folder):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for i, output in enumerate(self.outputs):
            if len(self.boxes_in_image[i])>0:                
                # print(f"Saving image {i} to {output_folder}/{i}.jpg")
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


