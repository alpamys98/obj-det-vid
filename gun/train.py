from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(data="./datasets/gun.yaml", imgsz=800, batch=8, epochs=50, plots=True)