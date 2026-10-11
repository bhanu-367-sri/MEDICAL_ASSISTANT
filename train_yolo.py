"""
Example training launcher for ONE specialist YOLO model.
First prepare a correctly labelled YOLO detection dataset and its data.yaml.
Run:
    python train_yolo.py --data datasets/fracture/data.yaml --name fracture_demo
This is a template, not a guarantee that the resulting model is clinically valid.
"""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--name", default="medical_experiment")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/medical",
        name=args.name,
        patience=10,
    )
    print("Training finished. Evaluate on a held-out test set before using the weights.")
    print("Best checkpoint is typically under runs/medical/<name>/weights/best.pt")

if __name__ == "__main__":
    main()
