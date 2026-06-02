import json
import re
import streamlit as st
from pathlib import Path

DATA_PATH = Path("/data1/lokesh/bep/data_gen/checkset_sharegpt_data_may23.json")

st.set_page_config(page_title="Sample Viewer", layout="wide")

@st.cache_data
def load_data():
    with open(DATA_PATH) as f:
        curr_data = json.load(f)
    return curr_data
    

data = load_data()
total = len(data)

if "idx" not in st.session_state:
    st.session_state.idx = 0

def go_prev():
    st.session_state.idx = max(0, st.session_state.idx - 1)

def go_next():
    st.session_state.idx = min(total - 1, st.session_state.idx + 1)

sample = data[st.session_state.idx]
messages = sample["messages"]
images = sample["images"]

human_msg = next((m["content"] for m in messages if m["role"] == "human"), "")
gpt_response = next((m["content"] for m in messages if m["role"] == "gpt"), "")

url_match = re.search(r"URL of this page is\s+(\S+)", human_msg)
url = url_match.group(1) if url_match else "N/A"

boxes_match = re.search(r"Boxes in the image are:\s*([\d,\s]+)\.", human_msg)
boxes = boxes_match.group(1).strip() if boxes_match else "N/A"

colors_match = re.search(r"Each box is highlighted with these colors:\s*(\{.*?\})", human_msg)
colors = colors_match.group(1) if colors_match else "N/A"

folder_num = Path(images[0]).parent.name if images else "N/A"

# Header
st.title("Phishing Dataset Viewer")
col_nav1, col_counter, col_nav2 = st.columns([1, 2, 1])
with col_nav1:
    st.button("← Previous", on_click=go_prev, disabled=st.session_state.idx == 0, use_container_width=True)
with col_counter:
    st.markdown(f"<h3 style='text-align:center'>Sample {st.session_state.idx + 1} / {total} &nbsp;|&nbsp; Folder: {Path(data[st.session_state.idx]['images'][0]).parent.name}</h3>", unsafe_allow_html=True)
with col_nav2:
    st.button("Next →", on_click=go_next, disabled=st.session_state.idx == total - 1, use_container_width=True)

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Page Screenshot")
    main_image = images[0]
    if Path(main_image).exists():
        st.image(main_image, use_container_width=True)
    else:
        st.warning(f"Image not found: `{main_image}`")

with right:
    st.subheader("Sample Info")
    st.markdown(f"**Folder:** `{folder_num}`")
    st.markdown(f"**URL:** `{url}`")
    st.markdown(f"**Boxes:** `{boxes}`")
    st.markdown(f"**Colors:** `{colors}`")

    st.subheader("GPT Response")
    st.code(gpt_response, language="json")

    if len(images) > 1:
        st.subheader("Box Crops")
        crop_images = images[1:]
        box_nums = [b.strip() for b in boxes.split(",")] if boxes != "N/A" else []
        cols = st.columns(min(len(crop_images), 4))
        for i, crop_path in enumerate(crop_images):
            col = cols[i % 4]
            label = f"Box {box_nums[i]}" if i < len(box_nums) else f"Crop {i}"
            with col:
                if Path(crop_path).exists():
                    st.image(crop_path, caption=label, use_container_width=True)
                else:
                    st.warning(f"Missing: `{crop_path}`")
