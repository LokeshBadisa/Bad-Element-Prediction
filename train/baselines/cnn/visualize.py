"""
Streamlit app to browse wrong CNN predictions saved by inference.py.
Run: streamlit run visualize.py
"""
import json
from pathlib import Path

import streamlit as st
from PIL import Image

CHECKERS_DIR = Path('checkers')
LABEL_NAME = {0: 'benign', 1: 'malicious'}
LABEL_COLOR = {0: 'green', 1: 'red'}

st.set_page_config(page_title='CNN Error Viewer', layout='wide')
st.title('CNN Inference — Wrong Predictions')

results_path = CHECKERS_DIR / 'results.json'
if not results_path.exists():
    st.error(f'`{results_path}` not found. Run `python inference.py` first.')
    st.stop()

with open(results_path) as f:
    results = json.load(f)

# ---- sidebar: summary + filters ----
with st.sidebar:
    st.header('Summary')
    total = len(results)
    fp = sum(1 for r in results if r['actual'] == 0 and r['predicted'] == 1)
    fn = sum(1 for r in results if r['actual'] == 1 and r['predicted'] == 0)
    st.metric('Total wrong predictions', total)
    col_a, col_b = st.columns(2)
    col_a.metric('False Positives', fp, help='benign predicted as malicious')
    col_b.metric('False Negatives', fn, help='malicious predicted as benign')

    st.divider()
    st.header('Filters')
    error_filter = st.selectbox(
        'Error type',
        ['All', 'False Positive (benign → malicious)', 'False Negative (malicious → benign)'],
    )
    conf_min, conf_max = st.slider(
        'Confidence range', 0.0, 1.0, (0.0, 1.0), step=0.01
    )

# ---- apply filters ----
filtered = results
if error_filter.startswith('False Positive'):
    filtered = [r for r in results if r['actual'] == 0 and r['predicted'] == 1]
elif error_filter.startswith('False Negative'):
    filtered = [r for r in results if r['actual'] == 1 and r['predicted'] == 0]
filtered = [r for r in filtered if conf_min <= r['confidence'] <= conf_max]

if not filtered:
    st.info('No samples match the current filters.')
    st.stop()

# ---- index state ----
if 'idx' not in st.session_state:
    st.session_state.idx = 0
# clamp after filter change
st.session_state.idx = min(st.session_state.idx, len(filtered) - 1)

# ---- navigation bar ----
nav_left, nav_mid, search_col, nav_right = st.columns([1, 2, 1, 1])

with nav_left:
    if st.button('← Previous', disabled=st.session_state.idx == 0, use_container_width=True):
        st.session_state.idx -= 1
        st.rerun()

with nav_mid:
    st.markdown(
        f"<div style='text-align:center; padding-top:6px;'>"
        f"Sample <b>{st.session_state.idx + 1}</b> of <b>{len(filtered)}</b> wrong predictions"
        f"</div>",
        unsafe_allow_html=True,
    )

with search_col:
    jump = st.number_input(
        'Go to', min_value=1, max_value=len(filtered), value=None,
        placeholder='#', label_visibility='collapsed',
    )
    if jump is not None:
        st.session_state.idx = int(jump) - 1
        st.rerun()

with nav_right:
    if st.button('Next →', disabled=st.session_state.idx == len(filtered) - 1, use_container_width=True):
        st.session_state.idx += 1
        st.rerun()

st.divider()

# ---- current sample ----
r = filtered[st.session_state.idx]
sample_dir = CHECKERS_DIR / str(r['id'])
ctx_path = sample_dir / 'context.png'
crop_path = sample_dir / 'crop.png'

actual_name = LABEL_NAME[r['actual']]
pred_name = LABEL_NAME[r['predicted']]
error_tag = 'False Positive' if r['actual'] == 0 else 'False Negative'
conf = r['confidence']
actual_color = LABEL_COLOR[r['actual']]
pred_color = LABEL_COLOR[r['predicted']]

img_col, crop_col, meta_col = st.columns([4, 1, 1])

with img_col:
    if ctx_path.exists():
        st.image(Image.open(ctx_path), caption='Context (full page)', use_container_width=True)
    else:
        st.warning('context.png missing')

with crop_col:
    if crop_path.exists():
        st.image(Image.open(crop_path), caption='Box (crop)')
    else:
        st.warning('crop.png missing')

with meta_col:
    st.markdown(f'### #{r["id"]}')
    st.markdown(f'**Type:** {error_tag}')
    st.divider()
    st.markdown(f'**Actual:** :{actual_color}[**{actual_name}**]')
    st.markdown(f'**Predicted:** :{pred_color}[**{pred_name}**]')
    st.markdown(f'**Confidence:** `{conf:.3f}`')
    st.divider()
    st.markdown('**All box labels (this page)**')
    full_labels = r.get('full_labels', {})
    if full_labels:
        box_num = str(Path(r.get('crop_src', '')).stem)
        for k, v in sorted(full_labels.items(), key=lambda x: int(x[0])):
            color = 'red' if v == 'malicious' else 'green'
            marker = ' ◀' if k == box_num else ''
            st.markdown(f':{color}[{k}: **{v}**]{marker}')
    else:
        st.caption('not available')
    st.divider()
    st.markdown('**Source paths**')
    st.caption(r.get('context_src', '—'))
    st.caption(r.get('crop_src', '—'))
