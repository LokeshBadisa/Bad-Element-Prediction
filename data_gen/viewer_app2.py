import json
import re
import streamlit as st
from pathlib import Path

SSIM_JSON = Path("/data1/lokesh/bep/data_gen/ssim_clusters_may20.json")
GLR_DIR = Path("/data1/lokesh/bep/data_gen/GLR_json")
BASE_DIR = Path("/data1/lokesh/v2_zip_openphish_usable")

st.set_page_config(page_title="SSIM Cluster Viewer", layout="wide")


def extract_labels(content):
    json_blocks = re.findall(r'```json(.*?)```', content, re.DOTALL)
    for match in json_blocks:
        cleaned = re.sub(r'\s+', ' ', match).strip()
        try:
            D = json.loads(cleaned)
            if 'final_label' in D:
                return D['final_label']
        except Exception:
            pass
    return None


@st.cache_data
def load_data():
    ssim = json.load(open(SSIM_JSON))
    folders = list(ssim.keys())
    return ssim, folders


ssim_data, folders = load_data()
total = len(folders)

if "idx" not in st.session_state:
    st.session_state.idx = 0


def go_prev():
    st.session_state.idx = max(0, st.session_state.idx - 1)


def go_next():
    st.session_state.idx = min(total - 1, st.session_state.idx + 1)


folder = folders[st.session_state.idx]
cluster = ssim_data[folder]  # list: [folder_name, [similar, score], ...]
similar = cluster[1:]        # [[folder_name, score], ...]

# Load labels from GLR_json
glr_path = GLR_DIR / f"{folder}.json"
label_map = {}
if glr_path.exists():
    raw = json.load(open(glr_path))
    for k, v in raw.items():
        lbl = extract_labels(v)
        if lbl:
            label_map[k] = lbl

# Header nav
st.title("SSIM Cluster Viewer")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.button("← Previous", on_click=go_prev, disabled=st.session_state.idx == 0, use_container_width=True)
with col2:
    st.markdown(
        f"<h3 style='text-align:center'>Folder {folder} &nbsp;|&nbsp; {st.session_state.idx + 1} / {total}</h3>",
        unsafe_allow_html=True,
    )
with col3:
    st.button("Next →", on_click=go_next, disabled=st.session_state.idx == total - 1, use_container_width=True)

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("final_boxes.jpg")
    img_path = BASE_DIR / folder / "final_boxes.jpg"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning(f"Image not found: `{img_path}`")

with right:
    st.subheader("Labels")
    if label_map:
        label_counts: dict[str, int] = {}
        for k in sorted(label_map.keys()):
            lbl = label_map[k]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            st.markdown(f"- `{k}` → **{lbl}**")
        st.divider()
        st.markdown("**Summary:**")
        for lbl, cnt in sorted(label_counts.items()):
            st.markdown(f"- {lbl}: {cnt}")
    else:
        st.info("No GLR labels found for this folder.")

    st.subheader("SSIM Cluster")
    st.markdown(f"**Representative:** `{folder}` ({len(similar)} similar)")
    if similar:
        for entry in similar:
            sim_folder, score = entry[0], entry[1]
            st.markdown(f"- `{sim_folder}` — {score:.4f}")
    else:
        st.info("No similar folders in cluster.")
