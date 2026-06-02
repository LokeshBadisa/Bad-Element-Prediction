import json
import re
from pathlib import Path

import streamlit as st
from PIL import Image

DATASET_PATH = "/data1/lokesh/bep/data_gen/testset_sharegpt_data_may23.json"
DEFAULT_PREDICTIONS = "/data1/lokesh/LlamaFactory/generated_predictions_1epoch_all_samples_may23_new.jsonl"
FALLBACK_PREDICTIONS = "/data1/lokesh/LlamaFactory/generated_predictions_all_samples_may23.jsonl"

st.set_page_config(page_title="BEP Inference Viewer", layout="wide")


def parse_label(label_str: str) -> dict[str, str]:
    match = re.search(r"\{([^}]+)\}", label_str)
    if not match:
        return {}
    content = match.group(1)
    result = {}
    for pair in content.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, v = pair.split(":", 1)
            result[k.strip()] = v.strip().lower()
    return result


@st.cache_data
def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


@st.cache_data
def load_predictions(path: str) -> list[dict]:
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def sample_accuracy(actual: dict, predicted: dict) -> tuple[int, int]:
    correct = sum(1 for k, v in actual.items() if predicted.get(k) == v)
    return correct, len(actual)


@st.cache_data
def get_error_indices(pred_path: str) -> list[int]:
    dataset = load_dataset()
    predictions = load_predictions(pred_path)
    indices = []
    for i, (sample, pred_entry) in enumerate(zip(dataset, predictions)):
        gpt_response = next((m["content"] for m in sample["messages"] if m["role"] == "gpt"), "")
        actual = parse_label(gpt_response)
        predicted = parse_label(pred_entry["predict"])
        correct, total = sample_accuracy(actual, predicted)
        if total > 0 and correct < total:
            indices.append(i)
    return indices


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")

pred_path = st.sidebar.text_input(
    "Predictions file",
    value=DEFAULT_PREDICTIONS if Path(DEFAULT_PREDICTIONS).exists() else FALLBACK_PREDICTIONS,
)

if not Path(pred_path).exists():
    st.sidebar.error(f"File not found:\n{pred_path}")
    st.info("Predictions file not found. Run inference first or enter a valid path in the sidebar.")
    st.stop()

dataset = load_dataset()
predictions = load_predictions(pred_path)
error_indices = get_error_indices(pred_path)

if not error_indices:
    st.success("All samples have 100% accuracy!")
    st.stop()

# ── Navigation ───────────────────────────────────────────────────────────────
if "pos" not in st.session_state:
    st.session_state.pos = 0


def go_prev():
    st.session_state.pos = max(0, st.session_state.pos - 1)


def go_next():
    st.session_state.pos = min(len(error_indices) - 1, st.session_state.pos + 1)


st.title("BEP Inference Viewer — Errors Only")

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 2, 3])
with nav_col1:
    st.button("← Prev", on_click=go_prev, use_container_width=True)
with nav_col2:
    st.button("Next →", on_click=go_next, use_container_width=True)
with nav_col3:
    jump = st.number_input(
        "Go to error #",
        min_value=0,
        max_value=len(error_indices) - 1,
        value=st.session_state.pos,
        step=1,
        key="jump_input",
    )
    if jump != st.session_state.pos:
        st.session_state.pos = jump
        st.rerun()
with nav_col4:
    orig_idx = error_indices[st.session_state.pos]
    st.markdown(
        f"**Error {st.session_state.pos + 1} / {len(error_indices)}** "
        f"&nbsp;·&nbsp; dataset index {orig_idx}"
    )

pos = st.session_state.pos
orig_idx = error_indices[pos]
sample = dataset[orig_idx]
pred_entry = predictions[orig_idx]

messages = sample["messages"]
images = sample["images"]

human_msg = next((m["content"] for m in messages if m["role"] == "human"), "")
gpt_response = next((m["content"] for m in messages if m["role"] == "gpt"), "")

actual_labels = parse_label(gpt_response)
predicted_labels = parse_label(pred_entry["predict"])

# ── Metadata ─────────────────────────────────────────────────────────────────
url_match = re.search(r"URL of this page is\s+(\S+?)\.", human_msg)
url = url_match.group(1) if url_match else "N/A"
folder_id = Path(images[0]).parent.name if images else "N/A"

correct, total_boxes = sample_accuracy(actual_labels, predicted_labels)
accuracy_pct = (correct / total_boxes * 100) if total_boxes else 0
acc_color = "orange" if accuracy_pct >= 50 else "red"

meta_col1, meta_col2, meta_col3 = st.columns([3, 1, 1])
with meta_col1:
    st.markdown(f"**URL:** {url}")
    st.markdown(f"**Folder:** `{folder_id}`")
with meta_col2:
    st.markdown(f"**Boxes:** {total_boxes}")
with meta_col3:
    st.markdown(f"**Accuracy:** :{acc_color}[{correct}/{total_boxes} ({accuracy_pct:.0f}%)]")

st.divider()

# ── Context image ─────────────────────────────────────────────────────────────
context_path = images[0]
crop_paths = images[1:]

st.subheader("Context Image (Full Screenshot)")
if Path(context_path).exists():
    ctx_img = Image.open(context_path)
    st.image(ctx_img, use_container_width=True)
else:
    st.warning(f"Context image not found: `{context_path}`")

st.divider()

# ── Raw labels ────────────────────────────────────────────────────────────────
label_col1, label_col2 = st.columns(2)
with label_col1:
    st.subheader("Actual Label")
    st.code(gpt_response, language=None)
with label_col2:
    st.subheader("Predicted Label")
    st.code(pred_entry["predict"], language=None)

st.divider()

# ── Box crops grid ────────────────────────────────────────────────────────────
st.subheader("Box Crops")

box_id_to_path: dict[str, str] = {}
for p in crop_paths:
    box_id = Path(p).stem
    box_id_to_path[box_id] = p


def sort_key(box_id: str):
    try:
        return int(box_id)
    except ValueError:
        return box_id


sorted_box_ids = sorted(box_id_to_path.keys(), key=sort_key)

COLS = 4
rows = [sorted_box_ids[i : i + COLS] for i in range(0, len(sorted_box_ids), COLS)]

for row_ids in rows:
    cols = st.columns(len(row_ids))
    for col, box_id in zip(cols, row_ids):
        crop_path = box_id_to_path[box_id]
        actual = actual_labels.get(box_id, "?")
        predicted = predicted_labels.get(box_id, "?")
        match = actual == predicted

        with col:
            st.markdown(f"**Box {box_id}**")
            if Path(crop_path).exists():
                img = Image.open(crop_path)
                st.image(img, use_container_width=True)
            else:
                st.warning("Image missing")

            if match:
                st.success(f"✓ {actual}")
            else:
                st.error(f"Actual: {actual}")
                st.warning(f"Pred:   {predicted}")
