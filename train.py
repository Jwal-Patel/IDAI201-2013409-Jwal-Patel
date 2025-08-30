import argparse
from ultralytics import YOLO
import torch

LABEL_MAP = {
    'helmet': 'Hardhat',
    'hardhat': 'Hardhat',
    'mask': 'Mask',
    'face_mask': 'Mask',
    'vest': 'Safety Vest',
    'safety_vest': 'Safety Vest'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='ppe_train')
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)

    # After training, run validation on your dataset
    try:
        res = model.val(data=args.data)
        print('Validation results:', res.metrics if hasattr(res, 'metrics') else res)
    except Exception as e:
        print('Validation step failed:', e)

if __name__ == "__main__":
    main()
