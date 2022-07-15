'''
https://learnopencv.com/edge-detection-using-opencv/

Edge Detection, bir görüntü içindeki nesnelerin veya bölgelerin sınırlarını (kenarlarını) belirlemek için 
kullanılan bir görüntü işleme tekniğidir.


'''
import cv2


img = cv2.imread('_resources/_images_must/edge_test.jpg')
# Girdi resmi görüntüleyin
cv2.imshow('Original', img)
cv2.waitKey(0)
 
# graycale e dönüştür
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Daha iyi kenar algılama için görüntüyü bulanıklaştırın
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
 

# Sobel Edge Detection
'''
Sobel(src, ddepth, dx, dy)

- src: girdi görüntüsüdür.
- ddepth: çıktı görüntüsünün kesinliğini belirtir.
- Eğer dx=1 ve dy=0 ise, x yönünde 1. türev görüntüsünü hesaplıyoruz.
- dx=1 ve dy=1 ise, 1. türev görüntüsünü her iki yönde de hesaplarız.
 
---Sobel Operatörü, piksel yoğunluğundaki ani değişikliklerle işaretlenen kenarları algılar.
'''

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # X ekseninde algılama
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Y ekseninde algılama
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Birleşik X ve Y de algılama

# Sobel Edge Detection sonuçlarını görüntüleyin
cv2.imshow('Original', img)
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)
 
# Canny Edge Detection
'''
Canny(image, threshold1, threshold2)

- image: girdi görüntüsüdür.
- threshold1: minimum thresh değeridir.
- threshold2: maximum thresh değeridir.


---Canny Edge Detection, çok sağlam ve esnek bir yöntemdir.
Algoritmanın kendisi, bir görüntüden kenarları çıkarmak için üç aşamalı bir süreci takip eder. Buna gürültüyü azaltmak için
gerekli bir ön işleme adımı olan görüntü bulanıklaştırma da eklenirse, onu aşağıdakileri içeren dört aşamalı bir süreç 
haline getirir:

--Gürültü Azaltma: Ham görüntü pikselleri genellikle gürültülü kenarlara yol açabilir, bu nedenle kenarları hesaplamadan önce 
paraziti azaltmak önemlidir Canny Edge Detection'da, istenmeyen kenarlara yol açabilecek gereksiz ayrıntıları ortadan 
kaldırmak veya en aza indirmek için bir Gauss bulanıklık filtresi kullanılır.

--Görüntünün Yoğunluk Gradyanını Hesaplama: Görüntü düzleştirildiğinde (bulanıklaştırıldığında), hem yatay hem de dikey 
olarak bir Sobel çekirdeği ile filtrelenir. Bu filtreleme işlemlerinden elde edilen sonuçlar ile gradyan büyüklüğü ve yönü hesaplanır.
daha sonra gradyan yönü  en yakın 45 derecelik açıya yuvarlanır.

--Yanlış Kenarların Bastırılması: Gürültüyü azalttıktan ve yoğunluk gradyanını hesapladıktan sonra, bu adımdaki algoritma, 
istenmeyen pikselleri (bir kenar oluşturmayabilir) filtrelemek için non-maximum suppression adı verilen bir teknik kullanır.
Bunu başarmak için her piksel, pozitif ve negatif gradyan yönünde komşu pikselleriyle karşılaştırılır. Mevcut pikselin gradyan 
büyüklüğü komşu piksellerden büyükse değişmeden bırakılır. Aksi takdirde, mevcut pikselin büyüklüğü sıfıra ayarlanır.

--Histerezis Eşiği: Canny Edge Detection'ın bu son adımında, gradyan büyüklükleri, biri diğerinden daha küçük olan iki eşik değeriyle karşılaştırılır. 

-Gradyan büyüklük değeri, daha büyük eşik değerinden yüksekse, bu pikseller güçlü kenarlarla ilişkilendirilir ve son kenar haritasına dahil edilir.
-Gradyan büyüklük değerleri daha küçük eşik değerinden düşükse pikseller bastırılır ve son kenar haritasından çıkarılır.
-Gradyan büyüklükleri bu iki eşik arasında kalan diğer tüm pikseller 'zayıf' kenarlar olarak işaretlenir (yani, son kenar haritasına dahil edilmeye aday olurlar). 
-"Zayıf" pikseller, güçlü kenarlarla ilişkili olanlara bağlanırsa, bunlar da son kenar haritasına dahil edilir.
'''

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

# Canny Edge Detection sonuçlarını görüntüleyin
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
 

cv2.destroyAllWindows()
