#%%
'''
images

https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/

!!!!!!!!!!! Test Edilmedi !!!!!!!!!!!!!!!
'''

import cv2
import numpy as np

'''
Yeni bir satırla ayrılmış tüm sınıf adlarını içeren object_detection_classes_coco.txt dosyasını okuyoruz. Her sınıf 
adını class_names listesinde saklıyoruz.
'''
# COCO class adlarını yükle
with open('object_detection_classes_coco.txt', 'r') as f:
   class_names = f.read().split('\n')
  
# class ların her biri için farklı bir renk dizisi alın
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))


# sinir ağı modelini yükle
'''
model = cv2.dnn.readNet(model=, config=, framework=)

model: Bu, önceden eğitilmiş ağırlıklar dosyasının yoludur. Bizim durumumuzda, önceden eğitilmiş Caffe modelidir.
config: Bu, model yapılandırma dosyasının yoludur ve bu durumda Caffe modelinin .prototxt dosyasıdır.
framework: Son olarak, modelleri yüklediğimiz çerçeve adını sağlamamız gerekiyor. Bizim için Caffe çerçevesidir.

DNN modülü, readNet() fonksiyonunun yanı sıra, framework argümanı sağlamak zorunda olmadığımız belirli çerçevelerden 
modelleri yüklemek için işlevler de sağlar. Aşağıdakiler bu işlevlerdir.


- readNetFromCaffe(): Bu, önceden eğitilmiş Caffe modellerini yüklemek için kullanılır ve iki argümanı kabul eder. 
Bunlar prototxt dosyasının yolu ve Caffe model dosyasının yoludur.

- readNetFromTensorflow(): TensorFlow önceden eğitilmiş modelleri doğrudan yüklemek için bu işlevi kullanabiliriz. 
Bu aynı zamanda iki argümanı da kabul eder. Biri frozen model grafiğinin yolu, diğeri ise model mimarisi protobuf 
metin dosyasının yoludur.

- readNetFromTorch(): Bunu torch.save() işlevi kullanılarak kaydedilen Torch ve PyTorch modellerini yüklemek için 
kullanabiliriz. Model yolunu argüman olarak sağlamamız gerekiyor.

- readNetFromDarknet(): Bu, DarkNet çerçevesi kullanılarak eğitilen modelleri yüklemek için kullanılır. Burada da 
iki argüman sunmamız gerekiyor. Biri model ağırlıklarının yolu, diğeri ise model config dosyasının yoludur.

- readNetFromONNX(): Bunu ONNX modellerini yüklemek için kullanabiliriz ve sadece ONNX model dosyasının yolunu 
sağlamamız gerekir.
'''

model = cv2.dnn.readNet(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', framework='TensorFlow')


'''
Nesne tespiti için blobFromImage() fonksiyonunda biraz farklı argüman değerleri kullanıyoruz.

- size: SSD modellerinin genellikle tüm çerçevelerden beklediği giriş boyutu olarak 300x300 olarak belirtiyoruz.
TensorFlow için de aynıdır.
- swapRB: Bu sefer swapRB argümanı da kullanıyoruz. Genel olarak, OpenCV görüntüyü BGR formatında okur ve nesne 
algılama için modeller, girişin genellikle RGB formatında olmasını bekler. Bu nedenle, swapRB argümanü görüntünün 
R ve B kanallarını değiştirerek onu RGB formatına dönüştürür.

Daha sonra blobu MobileNet SSD modeline ayarlıyoruz ve onu forward() fonksiyonunu kullanarak iletiyoruz.

Sahip olduğumuz output, aşağıdaki gibi yapılandırılmıştır:

[[[[0.00000000e+00 1.00000000e+00 9.72869813e-01 2.06566155e-02 1.11088693e-01 2.40461200e-01 7.53399074e-01]]]]

Burada 1. dizin konumu, 1'den 80'e kadar olabilen sınıf etiketini içerir. 
2. dizin konumu, güven puanını içerir. Bu bir olasılık puanı değil, modelin tespit ettiği sınıfa ait nesneye olan güvenidir.
Son dört değerden ilk ikisi x, y bounding box koordinatlarıdır ve son ikisi bounding box genişliği ve yüksekliğidir.
'''

# diskten görüntüyü oku
image = cv2.imread('../../input/image_2.jpg')
image_height, image_width, _ = image.shape

# görüntüden blob oluştur
blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)

# blobu modele göre ayarla
model.setInput(blob)

# algılamayı gerçekleştirmek için modelden ileri geçiş
output = model.forward()

'''
output içindeki algılamalar üzerinde döngü oluşturmaya ve algılanan nesnelerin her birinin çevresine bounding box lar 
çizmeye hazırız. Algılamalar üzerinde döngü yapmak için kod aşağıdadır.
'''

# algılamanın her biri üzerinde döngü
for detection in output[0, 0, :, :]:
   # algılamanın confidence değerini çıkarın
   confidence = detection[2]
   # yalnızca algılama güveni belirlenen değerden yukarıdaysa sınırlayıcı kutular çizin 
   # belirli bir eşik, yoksa atla
   if confidence > .4:
       # class id yi al
       class_id = detection[1]
       # class id yi class ile eşleştirin
       class_name = class_names[int(class_id)-1]
       color = COLORS[int(class_id)]
       # bounding box koordinatlarını al
       box_x = detection[3] * image_width
       box_y = detection[4] * image_height
       # bounding box ın width ve height ını al
       box_width = detection[5] * image_width
       box_height = detection[6] * image_height
       # algılanan her nesnenin etrafına bir dikdörtgen çizin
       cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
       # FPS metnini çerçevenin üstüne koy
       cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
  
cv2.imshow('image', image)
cv2.imwrite('image_result.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
'''
Videos
'''
import cv2
import time
import numpy as np
  
# COCO class adlarını yükle
with open('object_detection_classes_coco.txt', 'r') as f:
   class_names = f.read().split('\n')
  
# class ların her biri için farklı bir renk dizisi alın
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
  
# DNN modelini yükle
model = cv2.dnn.readNet(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',framework='TensorFlow')

# video capture nesnesi oluştur
cap = cv2.VideoCapture('../../input/video_1.mp4')

# videoların uygun şekilde kaydedilmesi için video karelerinin width ve height alın
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 'VideoWriter()' nesnesini yarat
out = cv2.VideoWriter('video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))


'''
Aşağıdaki kod bloğunda model, videoda döngüye alınacak kare kalmayıncaya kadar her karedeki nesneleri algılar. 
Dikkat edilmesi gereken bazı önemli noktalar:

- Algılamalardan önceki başlangıç zamanını start değişkende ve bitiş zamanını algılama sona erdikten sonra saklıyoruz.

- Aşağıdaki zaman değişkenleri, FPS'yi (Saniyedeki Kare Sayısı) hesaplamamıza yardımcı olur. 

- Kodun son kısmında, OpenCV DNN modülünü kullanarak MobileNet SSD modellerini çalıştırırken ne tür bir hız bekleyebileceğimize
dair bir fikir edinmek için hesaplanan FPS'yi mevcut çerçevenin üstüne de yazıyoruz.

- Son olarak, her bir kareyi ekranda görselleştiriyor ve bunları da diske kaydediyoruz.
'''
# videonun her karesindeki nesneleri algıla
while cap.isOpened():
   ret, frame = cap.read()
   if ret:
       image = frame
       image_height, image_width, _ = image.shape
       # görüntüden blob oluştur
       blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
       # FPS'yi hesaplamak için başlangıç ​​zamanı
       start = time.time()
       model.setInput(blob)
       output = model.forward()      

       # tespitten sonra bitiş zamanı
       end = time.time()

       # mevcut kare algılama için FPS'yi hesapla
       fps = 1 / (end-start)

       # algılamaların her biri üzerinde döngü
       for detection in output[0, 0, :, :]:
           # algılamanın confidence ını hesaplayın
           confidence = detection[2]
           # yalnızca algılama güveni belirlenen değerden yukarıdaysa sınırlayıcı kutular çizin 
           # belirli bir eşik, yoksa atla
           if confidence > .4:
               # class id yi al
               class_id = detection[1]
               # class id ile class ı eşleştirin
               class_name = class_names[int(class_id)-1]
               color = COLORS[int(class_id)]
               # bounding box koordinatlarını al
               box_x = detection[3] * image_width
               box_y = detection[4] * image_height
               # bounding box width ve height ı al
               box_width = detection[5] * image_width
               box_height = detection[6] * image_height
               # algılanan her nesnenin etrafına bir dikdörtgen çizin
               cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
               # algılanan nesneye sınıf adı metnini koy
               cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
               # FPS metnini çerçevenin üstüne koy
               cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
       cv2.imshow('image', image)
       out.write(image)
       if cv2.waitKey(10) & 0xFF == ord('q'):
           break
   else:
       break
  
cap.release()
cv2.destroyAllWindows()