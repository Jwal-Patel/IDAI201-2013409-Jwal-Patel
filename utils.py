import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Compliance logic
# Given a person bbox and list of PPE detections, decide which PPE items are present
# Person bbox format: [x1,y1,x2,y2]
# PPE detection entries: dict with keys: 'label' (str), 'box' [x1,y1,x2,y2], 'conf'

def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2, (y1+y2)/2)


def is_center_inside(person_box, box):
    cx,cy = box_center(box)
    x1,y1,x2,y2 = person_box
    return (cx >= x1) and (cx <= x2) and (cy >= y1) and (cy <= y2)


def classify_compliance(ppe_present_set):
    required = set(['Hardhat', 'Mask', 'Safety Vest'])
    count = len(required & ppe_present_set)
    if count == 3:
        return 'Green'
    elif count > 0:
        return 'Yellow'
    else:
        return 'Red'


def compliance_score(ppe_present_set):
    return len(ppe_present_set) / 3.0

# Drawing utilities

def draw_boxes(image_np, detections, class_map=None):
    # image_np: OpenCV BGR image
    img = image_np.copy()
    for det in detections:
        x1,y1,x2,y2 = map(int, det['box'])
        label = det.get('label','')
        conf = det.get('conf',0)
        cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        txt = f"{label} {conf:.2f}"
        cv2.putText(img, txt, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
    return img


def pil_from_cv2(cv2_img):

    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
