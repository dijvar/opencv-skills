from ultralytics import YOLO
from PIL import Image
import cv2
import torch


model = YOLO("D:/orhan/Belgeler/GitHub/opencv-skills/runs/detect/yolov8x_custom_imgsz_1024/weights/best.pt")
model = YOLO("D:/orhan/Belgeler/GitHub/opencv-skills/runs/detect/yolov8x_custom_bs_22/weights/best.pt")


print('cuda_avail:', torch.cuda.is_available())
print('cuda_device:', torch.cuda.device_count())



# Bir video nesnesi oluşturulur, bu durumda videoyu bir dosyadan okuyoruz
'''
vid_capture = cv2.VideoCapture(0) web kamerası
vid_capture = cv2.VideoCapture(1) normal kamera"
vid_capture = cv2.VideoCapture('Resources/Image_sequence/Cars%04d.jpg') >>> (Cars0001.jpg, Cars0002.jpg, Cars0003.jpg,...)
gibi bir görüntü dizinini okumaya yarar 
'''
vid_capture = cv2.VideoCapture('D:/orhan/Belgeler/Datasets/airborne_object_tracking/airborne-detection-starter-kit/data/part1/Images/006345553ed64e77a52d94035e1e747a/006345553ed64e77a52d94035e1e747a.mp4')

height, width = 2048 , 2448 

if (vid_capture.isOpened() == False):
  print("Error opening the video file")

# fps ve kare sayısı okunur
else:
  # Kare hızı bilgilerini almak 
  # 5'i CAP_PROP_FPS ile de değiştirebilirsiniz, bunlar numaralandırmadır
  fps = vid_capture.get(5)
  print('Frames per second : {} FPS'.format(fps))

 # Kare sayısını almak
 # 7'yi CAP_PROP_FRAME_COUNT ile de değiştirebilirsiniz, bunlar numaralandırmadır
  frame_count = vid_capture.get(7)
  print('Frame count : ', frame_count)

while(vid_capture.isOpened()):
  # vid_capture.read() bir tuple döndürür, ilk eleman bool ve ikincisi frame
  ret, frame = vid_capture.read()
  if ret == True:

    frame = cv2.resize(frame, (int(width/2),int(height/2)))


    results = model.predict(source=frame, save=True, show=True, device=0)

    cv2.imshow('Frame',frame)

    # waitKey() pencereyi kapatmak için bir tuşa basılmasını bekler ve 20 milisaniye cinsindendir
    key = cv2.waitKey(20)

    if key == ord('q'):
      break
  else:
    break

# Video capture nesnesini serbest bırakın
vid_capture.release()
cv2.destroyAllWindows()


vid_capture = cv2.VideoCapture('Resources/Videos/dog.mp4')

# get() kullanarak frame boyutu bilgilerini alın
frame_width = int(vid_capture.get(3)) # 3 >>> (CAP_PROP_FRAME_WIDTH)
frame_height = int(vid_capture.get(4)) # 4 >>> (CAP_PROP_FRAME_HEIGHT)
frame_size = (frame_width,frame_height)

# Video yazıcı nesnesini oluşturulur
'''
VideoWriter(filename, apiPreference, fourcc, fps, frameSize[, isColor])

filename: çıkış dosyasının yol adı
apiPreference: API backends tanımlayıcısı
fourcc: Kareleri sıkıştırmak için kullanılan 4 karakterli codec kodu 
AVI: cv2.VideoWriter_fourcc('M','J','P','G')
MP4: cv2.VideoWriter_fourcc(*'XVID')

fps: Oluşturulan video akışının kare hızı
frame_size: Video karelerinin boyutu
isColor: Sıfır değilse, kodlayıcı renkli çerçeveleri bekleyecek ve kodlayacaktır. 
Aksi takdirde gri tonlamalı çerçevelerle çalışacaktır (flag şu anda yalnızca Windows'ta desteklenmektedir).
'''
output = cv2.VideoWriter('_outputs/output_dog_video_file.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)

while(vid_capture.isOpened()):
    # vid_capture.read() bir tuple döndürür, ilk eleman bool ve ikincisi frame
    ret, frame = vid_capture.read()
    if ret == True:
           # Write the frame to the output files
           output.write(frame)
    else:
        print('Stream disconnected')
        break

