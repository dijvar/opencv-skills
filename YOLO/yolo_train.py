from ultralytics import YOLO
import torch

print('cuda_avail:', torch.cuda.is_available())
print('cuda_device:', torch.cuda.device_count())

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from scratch
#model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data="D:/orhan/Belgeler/Datasets/airborne_object_tracking/yolo_dataset/custom.yaml", epochs=2, imgsz=640)