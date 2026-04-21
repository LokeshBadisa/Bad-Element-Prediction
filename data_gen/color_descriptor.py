import colorsys
 
 
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
 
 
def rgb_to_hsl(r: int, g: int, b: int) -> tuple[int, int, int]:
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return round(h * 360), round(s * 100), round(l * 100)
 
 
def hue_name(h: int) -> str:
    ranges = [
        (15,  "red"),
        (30,  "red-orange"),
        (45,  "orange"),
        (65,  "yellow"),
        (80,  "yellow-green"),
        (150, "green"),
        (165, "green-cyan"),
        (195, "cyan"),
        (225, "blue"),
        (255, "blue-violet"),
        (285, "violet"),
        (315, "magenta"),
        (345, "pink"),
        (360, "red"),
    ]
    for threshold, name in ranges:
        if h < threshold:
            return name
    return "red"
 
 
def color_name(h: int, s: int, l: int) -> str:
    if s < 8:
        if l < 15:  return "black"
        if l < 35:  return "dark gray"
        if l < 60:  return "gray"
        if l < 85:  return "light gray"
        return "white"
 
    prefix = ""
    if l < 25:   prefix = "dark "
    elif l > 75: prefix = "light "
    elif l < 40: prefix = "deep "
    elif l > 60: prefix = "pale "
 
    sat_prefix = ""
    if s < 30:   sat_prefix = "grayish "
    elif s > 80: sat_prefix = "vivid "
 
    return (prefix + sat_prefix + hue_name(h)).strip()
 
 
 
 
def describe_color(rgb) -> str:
    r, g, b = (round(c * 255) for c in rgb)
    h, s, l = rgb_to_hsl(r, g, b)
 
    name = color_name(h, s, l)  # already 2-3 words max
 
    return name