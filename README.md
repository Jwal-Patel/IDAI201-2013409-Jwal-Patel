# PPE Detection System (YOLOv8 + Streamlit)

Detect Hardhat, Mask, and Safety Vest on construction workers and classify compliance per worker (Green/Yellow/Red).

## What you get
- `train.py` — YOLOv8 training script (Colab compatible)
- `data.yaml` — example dataset config for YOLO training
- `app.py` — Streamlit app for inference and dashboard
- `utils.py` — helper functions (compliance logic, visualization)
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Quick setup (Google Colab training)
1. Upload your dataset in YOLO format (images + labels). Example directory structure:

```
/dataset
  /train
    images/
    labels/
  /val
    images/
    labels/
  data.yaml
```

2. Open Google Colab and mount Google Drive.
3. Install dependencies: `pip install -U ultralytics opencv-python streamlit pandas altair torch torchvision pillow`
4. Copy `train.py` to Colab and run: `!python train.py --data /path/to/data.yaml --epochs 50 --imgsz 640 --weights yolov8n.pt` (adjust epochs and weights as needed)
5. After training, download `runs/detect/train/weights/best.pt` (or `best.pt`) and place it alongside `app.py` as `ppe_weights.pt`.

## Quick setup (local inference / Streamlit)
1. Create a virtual env, install `requirements.txt`.
2. Put your trained PPE model weights as `ppe_weights.pt` in project root.
3. Run: `streamlit run app.py`.

## Deployment (Streamlit Cloud)
1. Push repository to GitHub.
2. Sign in to Streamlit Cloud and connect the repo.
3. Set up `PPE_WEIGHTS` secret with a URL to download the weights on startup (optional). Or commit small `yolov8n.pt` replaced weights (not recommended).

"""

---