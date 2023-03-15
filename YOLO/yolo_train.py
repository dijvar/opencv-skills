from ultralytics import YOLO
import torch


if __name__ == '__main__':
    print('cuda_avail:', torch.cuda.is_available())
    print('cuda_device:', torch.cuda.device_count())

    data = "D:/orhan/Belgeler/Datasets/airborne_object_tracking/yolo_dataset/custom.yaml"
    last_weights = "D:/orhan/Belgeler/GitHub/opencv-skills/runs/detect/yolov8x_custom_bs_22/weights/best.pt"


    # Load a model
    # model = YOLO("YOLO/yolov8x_custom.yaml")  # build a new model from scratch
    model = YOLO(last_weights)  # load a pretrained model (recommended for training)

    # Train the model
    

    results = model.train(
    # resume=True,
    data=data,
    imgsz=640,
    epochs=50,
    batch=2,
    name='yolov8x_custom_imgsz_test',
    device=0,
    pretrained=True
    )

    #model.train(data=data, epochs=2, imgsz=640)