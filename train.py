from ultralytics import YOLO
import yaml

def train():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    model = YOLO("yolov8n.pt")  # or yolov8s.pt for larger model
    results = model.train(
        data=cfg["data"]["dataset_yaml"],
        epochs=cfg["train"]["epochs"],
        imgsz=cfg["data"]["img_size"],
        batch=cfg["data"]["batch_size"],
        lr0=cfg["train"]["lr0"],
        patience=cfg["train"]["patience"],
        augment=cfg["train"]["augment"],
        project=cfg["train"]["project"],
        name=cfg["train"]["name"],
        device=0  # GPU id, use 'cpu' if no GPU
    )
    model.save("models/best.pt")
    print("Training completed. Model saved to models/best.pt")

if __name__ == "__main__":
    train()