'''
https://learnopencv.com/color-spaces-in-opencv-cpp-python/

'''

import cv2
import numpy as np
import matplotlib as plt

bright = cv2.imread('_resources/_images_must/cube_bright.jpg')
dark = cv2.imread('_resources/_images_must/cube_dark.jpg')
cv2.imshow("Original Bright", bright)
cv2.imshow("Original Dark", dark)
cv2.waitKey()

# <RGB>
'''Renklerin Kırmızı, Yeşil ve Mavi değerlerinin doğrusal bir kombinasyonu ile elde edildiği bir katkılı renk uzayıdır.
Üç kanal, yüzeye çarpan ışık miktarı ile ilişkilidir.
'''



# <LAB>
'''
L - Hafiflik (Yoğunluk).
a - Yeşilden Macentaya değişen renk bileşeni.
b - Maviden Sarıya değişen renk bileşeni.

Lab renk uzayı, RGB renk uzayından oldukça farklıdır. RGB renk uzayında renk bilgisi üç kanala ayrılır 
ancak aynı üç kanal aynı zamanda parlaklık bilgisini de kodlar. Öte yandan Lab renk uzayında L kanalı renk 
bilgisinden bağımsızdır ve yalnızca parlaklığı kodlar. Diğer iki kanal rengi kodlar.

Ayrıca aşağıdaki özelliklere sahiptir:

- Rengi nasıl algıladığımıza yaklaşan algısal olarak tek tip renk uzayı.
- Cihazdan bağımsız (yakalama veya görüntüleme).
- Adobe Photoshop'ta yaygın olarak kullanılır.
- Karmaşık bir dönüşüm denklemi ile RGB renk uzayıyla ilişkilidir.
'''
brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
cv2.imshow("brightLAB", brightLAB)
cv2.imshow("darkLAB", darkLAB)
cv2.waitKey()

# <YCrCb>
'''
YCrCb renk alanı, RGB renk alanından türetilmiştir ve aşağıdaki üç bileşene sahiptir.

Y - Gama düzeltmesinden sonra RGB'den elde edilen parlaklık veya Luma bileşeni.
Cr = R - Y (kırmızı bileşen Luma'dan ne kadar uzakta).
Cb = B - Y (mavi bileşen Luma'dan ne kadar uzakta).
Bu renk uzayı aşağıdaki özelliklere sahiptir.

- Parlaklık ve krominans bileşenlerini farklı kanallara ayırır.
- Çoğunlukla TV İletimi için sıkıştırmada (Cr ve Cb bileşenlerinin) kullanılır.
- Cihaza bağlı.
'''
brightYCB = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)
cv2.imshow("brightYCB", brightYCB)
cv2.imshow("darkYCB", darkYCB)
cv2.waitKey()

# <HSV>
'''
HSV renk alanı aşağıdaki üç bileşene sahiptir

H - Ton (Baskın Dalga Boyu).
S - Doygunluk (Saflık / rengin tonları).
V - Değer (Yoğunluk).
Bazı özelliklerini sıralayalım.

En iyi yanı, rengi (H) tanımlamak için yalnızca bir kanal kullanması ve bu da rengi belirtmeyi çok sezgisel hale getirmesidir.
Cihaza bağlı.
'''
brightHSV = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
darkHSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
cv2.imshow("brightHSV", brightHSV)
cv2.imshow("darkHSV", darkHSV)
cv2.waitKey()


# bgr = [40 158 16]
# HSV = [65 229 158] 
# YCrCv = [102 67 93] 
# LAB = [145 71 177]  
bgr = [40, 158, 16] 
thresh = 40
 
minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
 
maskBGR = cv2.inRange(bright,minBGR,maxBGR)
resultBGR = cv2.bitwise_and(bright, bright, mask = maskBGR)
 
# convert 1D dizisini 3D'ye dönüştürün, ardından onu HSV'ye dönüştürün ve ilk elemanı alın
# bu yukarıdaki şekilde gösterildiği gibi olacaktır [65, 229, 158]
hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]
 
minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])
 
maskHSV = cv2.inRange(brightHSV, minHSV, maxHSV)
resultHSV = cv2.bitwise_and(brightHSV, brightHSV, mask = maskHSV)

# 1B diziyi 3B'ye dönüştürün, ardından onu YCrCb'ye dönüştürün ve ilk elemanı alın
ycb = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2YCrCb)[0][0]

minYCB = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
maxYCB = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh])

maskYCB = cv2.inRange(brightYCB, minYCB, maxYCB)
resultYCB = cv2.bitwise_and(brightYCB, brightYCB, mask = maskYCB)
 
#convert 1D dizisini 3D'ye dönüştürün, ardından onu LAB'ye dönüştürün ve ilk elemanı alın
lab = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2LAB)[0][0]

minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])
 
maskLAB = cv2.inRange(brightLAB, minLAB, maxLAB)
resultLAB = cv2.bitwise_and(brightLAB, brightLAB, mask = maskLAB)

cv2.imshow("Result BGR", resultBGR)
cv2.waitKey()
cv2.imshow("Result HSV", resultHSV)
cv2.waitKey()
cv2.imshow("Result YCB", resultYCB)
cv2.waitKey()
cv2.imshow("Output LAB", resultLAB)
cv2.waitKey()

cv2.destroyAllWindows()