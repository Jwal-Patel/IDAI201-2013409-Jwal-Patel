import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt
from utils import is_center_inside, classify_compliance, draw_boxes, pil_from_cv2


LABEL_MAP = {
    'helmet': 'Hardhat',
    'hardhat': 'Hardhat',
    'mask': 'Mask',
    'face_mask': 'Mask',
    'vest': 'Safety Vest',
    'safety_vest': 'Safety Vest'
}

st.set_page_config(layout='wide', page_title='PPE Detection')

st.title('PPE Detection & Compliance Dashboard')

# Sidebar controls
st.sidebar.header('Model Settings')
PPE_WEIGHTS = st.sidebar.text_input('PPE weights path', value='ppe_weights.pt')
CONF_THRESH = st.sidebar.slider('Confidence threshold', 0.0, 1.0, 0.35, 0.01)
IOU_THRESH = st.sidebar.slider('IoU threshold (NMS)', 0.0, 1.0, 0.45, 0.01)
USE_PERSON_MODEL = st.sidebar.checkbox('Use COCO person detector for grouping', value=True)
PERSON_CONF = st.sidebar.slider('Person model conf', 0.0, 1.0, 0.35, 0.01)

# Load models (cache)
@st.cache_resource
def load_models(ppe_weights, use_person):
    try:
        ppe_model = YOLO(ppe_weights)
    except Exception as e:
        st.error(f'Failed to load PPE model: {e}')
        ppe_model = None
    person_model = None
    if use_person:
        try:
            # yolov8n.pt pretrained on COCO will detect 'person' among other classes
            person_model = YOLO('yolov8n.pt')
        except Exception as e:
            st.warning('Could not load person detection model (yolov8n.pt). Make sure it is available or disable person grouping.')
            person_model = None
    return ppe_model, person_model

ppe_model, person_model = load_models(PPE_WEIGHTS, USE_PERSON_MODEL)

# Image upload or camera
st.sidebar.header('Input')
input_mode = st.sidebar.radio('Input mode', ['Upload Image','Camera (optional)'])

uploaded_file = None
if input_mode == 'Upload Image':
    uploaded_file = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
else:
    cam_file = st.camera_input('Take a photo')
    uploaded_file = cam_file

col1, col2 = st.columns((2,1))

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)[:, :, ::-1].copy()  # RGB to BGR for cv2

    # Run person detection if available
    persons = []
    if USE_PERSON_MODEL and person_model is not None:
        person_results = person_model.predict(source=np.array(image), conf=PERSON_CONF, iou=IOU_THRESH, classes=[0], verbose=False)
        # results[0].boxes.xyxyn or .boxes.xyxy
        try:
            boxes = person_results[0].boxes.xyxy.cpu().numpy()
            confs = person_results[0].boxes.conf.cpu().numpy()
            for b, c in zip(boxes, confs):
                x1,y1,x2,y2 = b.tolist()
                persons.append({'box':[x1,y1,x2,y2], 'conf':float(c)})
        except Exception:
            persons = []

    # Run PPE model
    ppe_dets = []
    if ppe_model is not None:
        res = ppe_model.predict(source=np.array(image), conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
        try:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy()
            labels = res[0].boxes.cls.cpu().numpy().astype(int)
            # map numeric labels to names — rely on model.names
            names = res[0].names if hasattr(res[0], 'names') else ppe_model.names
            for b,c,lab in zip(boxes, confs, labels):
                label_name = names[int(lab)]
                ppe_dets.append({'box':b.tolist(), 'conf':float(c), 'label':label_name})
        except Exception as e:
            st.warning('PPE model returned no boxes or failed to parse results: ' + str(e))

    # If no person detector, fallback: create one big person bbox and assign all PPE to one worker
    if not persons:
        # Create synthetic person bbox that covers image so we can at least count overall compliance
        h, w = img_np.shape[:2]
        persons = [{'box':[0,0,w,h], 'conf':1.0}]

    # Assign PPE to persons
    workers = []
    for p in persons:
        pbox = p['box']
        present = set()
        matched = []
        for d in ppe_dets:
            if is_center_inside(pbox, d['box']):
                normalized_label = LABEL_MAP.get(d['label'].lower(), d['label'])
                present.add(normalized_label)
                matched.append(d)
        # Add worker info
        workers.append({
            'person_box': pbox,
            'ppe_present': present,
            'ppe_dets': matched,
            'status': classify_compliance(present)
        })

    # Draw detections and worker boxes
    vis_img = img_np.copy()
    # Draw PPE boxes
    for d in ppe_dets:
        x1,y1,x2,y2 = map(int, d['box'])
        cv2.rectangle(vis_img, (x1,y1),(x2,y2),(0,255,0),2)
        normalized_label = LABEL_MAP.get(d['label'].lower(), d['label'])
        cv2.putText(vis_img, f"{normalized_label} {d['conf']:.2f}", (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

    # Draw person boxes and status
    for idx,wk in enumerate(workers):
        x1,y1,x2,y2 = map(int, wk['person_box'])
        status = wk['status']
        color = (0,200,0) if status=='Green' else ((0,200,200) if status=='Yellow' else (0,0,255))
        cv2.rectangle(vis_img, (x1,y1),(x2,y2), color, 3)
        label_text = f"Worker {idx+1}: {status}"
        if status != 'Green':
            required = set(['Hardhat','Mask','Safety Vest'])
            missing = sorted(list(required - wk['ppe_present']))
            if missing:
                label_text += f" (Missing: {', '.join(missing)})"
        cv2.putText(vis_img, label_text, (x1, max(25,y1+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

    # Convert to displayable image
    vis_pil = pil_from_cv2(vis_img)

    with col1:
        st.subheader('Detections')
        st.image(vis_pil, use_container_width=True)

    # Summary stats
    statuses = [w['status'] for w in workers]
    counts = pd.Series(statuses).value_counts().reindex(['Green','Yellow','Red']).fillna(0).astype(int)
    df = pd.DataFrame({'Status':counts.index, 'Count':counts.values})

    with col2:
        st.subheader('Compliance Summary')
        st.table(df)
        chart = alt.Chart(df).mark_bar().encode(x='Status', y='Count')
        st.altair_chart(chart, use_container_width=True)

        st.markdown('### Alerts')
        non_compliant = counts['Red'] if 'Red' in counts.index else 0
        partial = counts['Yellow'] if 'Yellow' in counts.index else 0
        if non_compliant > 0:
            st.error(f'{non_compliant} worker(s) WITHOUT any PPE detected!')
        if partial > 0:
            st.warning(f'{partial} worker(s) PARTIALLY compliant (missing 1 or 2 items).')
        if non_compliant==0 and partial==0:
            st.success('All detected workers are fully compliant.')

    # Option: show raw detections
    with st.expander('Show raw PPE detections'):
        st.write(ppe_dets)

else:
    st.info('Upload an image or take a photo to run PPE detection.')
