from ultralytics import YOLO
import torch


if __name__ == '__main__':
    print('cuda_avail:', torch.cuda.is_available())
    print('cuda_device:', torch.cuda.device_count())

    # Load a model
    model = YOLO("yolov8x.yaml")  # build a new model from scratch
    #model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

    # Train the model
    data = "D:/orhan/Belgeler/Datasets/airborne_object_tracking/yolo_dataset/custom.yaml"

    results = model.train(
    data=data,
    imgsz=640,
    epochs=30,
    batch=4,
    name='yolov8x_custom',
    device=0)

    #model.train(data=data, epochs=2, imgsz=640)