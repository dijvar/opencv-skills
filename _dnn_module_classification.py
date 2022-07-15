'''
OpenCV'yi en iyi bilgisayarla görme kitaplıklarından biri olarak biliyoruz. Ek olarak, derin öğrenme çıkarımını 
çalıştırma işlevlerine de sahiptir. En iyi yanı, çeşitli derin öğrenme işlevlerini gerçekleştirebileceğimiz farklı 
çerçevelerden farklı modellerin yüklenmesini desteklemektir. Farklı çerçevelerden modelleri destekleme özelliği, 
sürüm 3.3'ten beri OpenCV'nin bir parçası olmuştur.

OpenCV DNN modülünün en iyi yanlarından biri, Intel işlemciler için yüksek düzeyde optimize edilmiş olmasıdır. 
Nesne algılama ve görüntü bölütleme uygulamaları için gerçek zamanlı videolarda çıkarım yaparken iyi FPS elde 
edebiliyoruz. Belirli bir çerçeve kullanılarak önceden eğitilmiş bir model kullanırken genellikle DNN modülüyle 
daha yüksek FPS elde ederiz.

https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/

!!!!!!!!!!! Test Edilmedi !!!!!!!!!!!!!!!
'''

import cv2
import numpy as np

#  ImageNet class adlarını oku
with open('../../input/classification_classes_ILSVRC2012.txt', 'r') as f:
   image_net_names = f.read().split('\n')

# son sınıf adı (bir görüntü için birçok ImageNet adının yalnızca ilk sözcüğü)
class_names = [name.split(',')[0] for name in image_net_names]

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

model = cv2.dnn.readNet(model='../../input/DenseNet_121.caffemodel', config='../../input/DenseNet_121.prototxt', framework='Caffe')


'''
Her zamanki gibi imread() işlevini kullanarak görüntüyü diskten okuyacağız. Dikkat etmemiz gereken birkaç ayrıntı 
daha olduğunu unutmayın. DNN modülünü kullanarak yüklediğimiz önceden eğitilmiş modeller, doğrudan okunan görüntüyü 
girdi olarak almaz. Bundan önce bazı ön işlemler yapmamız gerekiyor. 

Resmi okurken, mevcut dizinden önce ve input klasörü içinde iki dizin olduğunu varsayıyoruz. Sonraki birkaç adım çok
önemlidir. Görüntüyü modele beslenmesi için doğru formatta hazırlayan bir blobFromImage() fonksiyonumuz var.

cv2.dnn.blobFromImage(image=, scalefactor=, size=, mean=)

- image: Bu, yukarıda imread() işlevini kullanarak okuduğumuz giriş görüntüsüdür.

- scalefactor: Bu değer, görüntüyü sağlanan değere göre ölçeklendirir. Varsayılan değer 1 dir, bu ölçeklendirme 
yapılmadığı anlamına gelir.

- size: Bu, görüntünün yeniden boyutlandırılacağı boyuttur. ImageNet veri setinde eğitilen çoğu sınıflandırma modeli
sadece bu boyutu beklediğinden, boyutu 224x224 olarak sağladık.

- mean: Ortalama argüman oldukça önemlidir. Bunlar aslında görüntünün RGB renk kanallarından çıkarılan ortalama 
değerlerdir. Bu, girişi normalleştirir ve son girişin farklı aydınlatma ölçeklerine değişmezliğini sağlar.

Burada dikkat edilmesi gereken bir şey daha var. Tüm derin öğrenme modelleri, toplu girdiler bekler. Ancak burada 
sadece bir görselimiz var. Yine de, burada elde ettiğimiz blob çıktısı [1, 3, 224, 224] idi. blobFromImage() fonksiyonu 
tarafından fazladan bir toplu iş boyutunun eklendiğini gözlemleyin. Bu, sinir ağı modeli için nihai ve doğru
giriş formatı olacaktır.
'''

# görüntüyü diskten yükleyin
image = cv2.imread('../../input/image_1.jpg')
# görüntüden blob oluştur
blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))

'''
Tahmin yapmak için iki adım vardır. 

Öncelikle diskten yüklediğimiz sinir ağı modelimize girdi blobunu ayarlamamız gerekiyor.
İkinci adım, bize tüm çıktıları veren blobu model boyunca ileriye doğru yaymak için forward() fonksiyonunu kullanmaktır.
Aşağıdaki kod bloğunda her iki adımı da gerçekleştiriyoruz.
'''

# sinir ağı için giriş bloğunu ayarla
model.setInput(blob)
# model aracılığıyla resim blogunu ileriye aktar
outputs = model.forward()

'''
Bir dizi olan outputs, tüm tahminleri tutar. Ancak çıktıları ve sınıf etiketlerini doğru bir şekilde görmeden önce 
tamamlamamız gereken birkaç ön işleme adımı var.

Şu anda outputs (1, 1000, 1, 1) şeklindedir ve sınıf etiketlerini olduğu gibi çıkarmak zordur. Bu nedenle, aşağıdaki 
kod bloğu outputs'u yeniden şekillendirir, ardından doğru sınıf etiketlerini kolayca alabilir ve etiket kimliğini 
sınıf adlarıyla eşleştirebiliriz.

Bunlardan en yüksek etiket indeksini çıkarıyoruz ve label_id de saklıyoruz. Ancak bu puanlar aslında olasılık puanları
değildir. Modelin en yüksek puanlı etiketi hangi olasılıkla tahmin ettiğini bilmek için softmax olasılıklarını elde 
etmemiz gerekiyor. 

Aşağıdaki kodda, np.exp(final_outputs)/np.sum(np.exp(final_outputs)) puanlarını kullanarak softmax olasılıklarına 
dönüştürüyoruz. Ardından, tahmin edilen puan yüzdesini elde etmek için en yüksek olasılık puanını 100 ile çarpıyoruz.

Son adımlar, resmin üstüne sınıf adını ve yüzdesini açıklamak olacaktır. Ardından görüntüyü görselleştirip sonucu 
diske de kaydediyoruz.
'''

final_outputs = outputs[0]

# tüm outputs ları 1D yap
final_outputs = final_outputs.reshape(1000, 1)

# class label larını al
label_id = np.argmax(final_outputs)

# output score ları softmax olasılıklarına dönüştürün
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))

# son en yüksek olasılığı elde et
final_prob = np.max(probs) * 100.

# maksimum confidence ı class label adlarıyla eşleştirin
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"

# class adı metnini resmin üstüne koyun
cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.imwrite('result_image.jpg', image)