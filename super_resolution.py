'''
Süper çözünürlük, görüntünün ayrıntılarını yükseltme veya iyileştirme sürecini ifade eder. Bir görüntünün boyutlarını 
arttırırken, ekstra piksellerin bir şekilde enterpolasyonu gerekir. Temel görüntü işleme teknikleri, büyütme sırasında 
çevreyi bağlam içine almadıkları için iyi sonuçlar vermezler. Derin öğrenme ve daha yakın zamanda GAN'lar burada 
kurtarmaya gelir ve çok daha iyi sonuçlar sağlar.

OpenCV şu anda görüntüleri yükseltmek için dört derin öğrenme algoritması seçeneği sunuyor. Dört yöntem şunlardır:

- EDSR
- ESPCN
- FSRCN
- LapSRN

İlk üç algoritmanın 2, 3 ve 4 kat yüksek ölçek oranı sunarken, son algoritmanın orijinal boyutunun 2, 4 ve 8 katı ölçek
oranı sunar.


https://learnopencv.com/super-resolution-in-opencv/

!!!!!!!!!!!!!! KOD CALISIYOR ANCAK TEST EDİLMEDİ !!!!!!!!!!!!!
'''
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('_resources/_photos/super_resolution.png')
plt.imshow(img[:,:,::-1])
plt.show()

# OpenCV logosunu kırpma
imgsa = img[:40,335:]
plt.imshow(imgsa[:,:,::-1])
plt.show()

# <EDSR>
'''
Toplu Normalleştirme katmanları olmadan ResNet stili bir mimari kullanılır. Bu katmanların, özelliklerin ağlarından 
menzil esnekliğinden kurtulduğunu ve performansı iyileştirdiğini buldular. Bu, daha iyi performansa sahip daha büyük bir 
model oluşturmalarını sağlar. Büyük modellerde bulunan kararsızlığa karşı koymak için, son evrişim katmanlarından sonra 
sabit ölçeklendirme katmanları yerleştirerek her artık blokta 0,1 faktörlü artık ölçekleme kullandılar. Ayrıca, artık 
bloklardan sonra ReLu aktivasyon katmanları kullanılmaz.

Mimari başlangıçta 2'lik bir ölçeklendirme faktörü için kullanılır. Daha sonra bu önceden eğitilmiş ağırlıklar, 3 ve 4'lük 
bir ölçeklendirme faktörü için eğitim yapılırken kullanılır. Bu sadece eğitimi hızlandırmakla kalmaz, aynı zamanda 
modellerin performansını da geliştirir.

'''

sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
path_EDSR = "EDSR_x4.pb"
 
sr.readModel(path_EDSR)
 
sr.setModel("edsr",4) # değeri ve örnekleme oranını vererek modeli ayarlayın
 
result_EDSR = sr.upsample(img) # giriş görüntüsünü yükselt
 
# Yeniden boyutlandırılmış resim
resized_4x4 = cv2.resize(img,dsize=None,fx=4,fy=4)
 
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)

# Orjinal resim
plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2)

# EDSR upscaled
plt.imshow(result_EDSR[:,:,::-1])
plt.subplot(1,3,3)

# OpenCV upscaled
plt.imshow(resized_4x4[:,:,::-1])
plt.show()

# <ESPCN>
'''
Bir bikübik filtre kullanarak düşük çözünürlüğü yükselttikten sonra süper çözünürlük gerçekleştirmek yerine, özellik 
haritalarını düşük çözünürlükte çıkarın ve sonucu elde etmek için karmaşık yükseltme filtreleri kullanın. Yükseltme 
katmanları yalnızca ağın sonunda dağıtılır. Bu, modelde meydana gelen karmaşık işlemlerin daha düşük boyutlarda 
gerçekleşmesini sağlar ve bu da özellikle diğer tekniklere göre daha hızlı olmasını sağlar.

ESPCN'nin temel yapısı SRCNN'den esinlenmiştir. Geleneksel evrişim katmanlarını kullanmak yerine, ters evrişim katmanları
gibi davranan alt piksel evrişim katmanları kullanılır. Alt piksel evrişim katmanı, yüksek çözünürlüklü haritayı üretmek 
için son katmanda kullanılır. Bununla birlikte Tanh aktivasyon fonksiyonunun standart ReLu fonksiyonundan çok daha iyi 
çalıştığını buldular.
'''

sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
path_ESPCN = "ESPCN_x3.pb"
 
sr.readModel(path_ESPCN)
 
sr.setModel("espcn",3) # değeri ve örnekleme oranını vererek modeli ayarlayın
 
result_ESPCN = sr.upsample(img) # giriş görüntüsünü yükselt
 
# Yeniden boyutlandırılmış resim 
resized_3x3 = cv2.resize(img,dsize=None,fx=3,fy=3)
 
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)

# Orjinal resim
plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2)

# SR upscaled
plt.imshow(result_ESPCN[:,:,::-1])
plt.subplot(1,3,3)

# OpenCV upscaled
plt.imshow(resized_3x3[:,:,::-1])
plt.show()



# <FSCNN>
'''
FSCNN ve ESPCN çok benzer kavramlara sahiptir. Her ikisi de SRCNN'den ilham alan temel yapılarına sahiptir ve başlangıçta 
enterpolasyon yapmak yerine hız için ölçek yükseltme katmanları kullanır. Ayrıca, girdi özelliği boyutunu bile küçültürler 
ve sonunda daha fazla eşleme katmanı kullanmadan önce daha küçük filtre boyutları kullanırlar, bu da modelin daha da küçük 
ve daha hızlı olmasına neden olur.

Mimari, filtre boyutu SRCNN'nin 9'undan 5'e düşürüldüğü evrişimli katmanlarla başlar. Girdi çözünürlüğünün kendisi çok 
büyük olabileceği ve çok zaman alabileceği için küçülen katmanlar uygulanır. Hesaplama maliyetini artırmayan 1x1 filtre 
boyutu kullanılır.

Yazarlar daha sonra, doğruluktan ödün vermeden modeli yavaşlatmada ayrılmaz bir rol oynayan doğrusal olmayan eşlemeyi 
azaltmaya odaklanıyorlar. Bu nedenle, birden fazla 3x3 filtre kullanırlar. Bir sonraki genişleyen bölüm, en sonunda üst 
örnekleme için evrişimsiz katmanları uygulamadan önce, küçülen bölümün tersidir. Aktivasyon fonksiyonu için PReLu kullanıldı.
'''

sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
path_FSCNN = "FSRCNN_x3.pb"
 
sr.readModel(path_FSCNN)
 
sr.setModel("fsrcnn",3) # değeri ve örnekleme oranını vererek modeli ayarlayın
 
result_FSCNN = sr.upsample(img) # giriş görüntüsünü yükselt
 
# Yeniden boyutlandırılmış resim
resized_3x3 = cv2.resize(img,dsize=None,fx=3,fy=3)
 
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)

# Orjinal resim
plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2)

# SR upscaled
plt.imshow(result_FSCNN[:,:,::-1])
plt.subplot(1,3,3)

# OpenCV upscaled
plt.imshow(resized_3x3[:,:,::-1])
plt.show()



# <LapSRN>
'''
LapSRN, başlangıç ve bitişte zıt ölçek yükseltme stratejileri arasında bir orta yol sunar. Sonuna kadar nazikçe yükseltmeyi 
öneriyor. Adı Laplacian piramitlerine dayanmaktadır ve mimari, temelde, düşük çözünürlüklü görüntüyü sonuna kadar 
yükselten bir piramit gibidir. Hız için parametre paylaşımına büyük ölçüde güvenilmektedir; ve tıpkı EDSR modelleri gibi, 
MS-LapSRN olarak adlandırılan farklı ölçekleri yeniden oluşturabilen tek bir model de önerdiler. Ancak burada 
sadece LapSRN'den bahsedilme.

Modeller iki daldan oluşur: özellik çıkarma ve bir görüntü rekonstrüksiyon dalı. Parametre paylaşımı farklı ölçekler 
arasında gerçekleşir, yani 4x, 2x modelinden vb. parametreleri kullanır. Bu, 2x ölçeklendirme için bir piramidin, 4x 
için iki piramidin ve 8x için üç piramidin kullanıldığı anlamına gelir! Bu kadar derin modeller yapmak, gradyan kaybolma 
problemlerinden muzdarip olabilecekleri anlamına gelir. Bu nedenle, farklı kaynak atlama bağlantıları ve paylaşılan 
kaynak bağlantıları gibi farklı türde yerel atlama bağlantıları denerler. Modelin kayıp işlevi için Charbonnier kaybı 
kullanılır ve toplu normalleştirme katmanları kullanılmaz.
'''

sr = cv2.dnn_superres.DnnSuperResImpl_create()
 
path_LapSRN = "LapSRN_x8.pb"
 
sr.readModel(path_LapSRN)

sr.setModel("lapsrn",8) # değeri ve örnekleme oranını vererek modeli ayarlayın
 
result_LapSRN = sr.upsample(img) # giriş görüntüsünü yükselt
 
# Yeniden boyutlandırılmış resim 
resized_8x8 = cv2.resize(img,dsize=None,fx=8,fy=8)
 
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)

# Orjinal resim
plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2)

# SR upscaled
plt.imshow(result_LapSRN[:,:,::-1])
plt.subplot(1,3,3)

# OpenCV upscaled
plt.imshow(resized_8x8[:,:,::-1])
plt.show()