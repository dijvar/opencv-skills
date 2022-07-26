'''
https://learnopencv.com/image-resizing-with-opencv/

'''
import cv2
import numpy as np
 

image = cv2.imread('_resources/_images_must/auto.jpg')
cv2.imshow('Original Image', image)

# resize() kullanarak yeniden boyutlandırma
'''
resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])

src: girdi görüntüsüdür.
dsize: Çıktı görüntüsünün istenen boyutudur, yeni bir yükseklik ve genişlik olabilir.
fx: Yatay eksen boyunca ölçek faktörü.
fy: Dikey eksen boyunca ölçek faktörü.
interpolation: Bize görüntüyü yeniden boyutlandırmak için farklı yöntemler seçeneği sunar.
'''

# <Genişlik ve Yükseklik Belirterek Yeniden Boyutlandırma>
'''
yeni bir genişlik ve yükseklik belirleyerek resmi yeniden boyutlandırma.
'''

# yeni genişlik ve yükseklik kullanarak resmi küçültelim
down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
 
# yeni genişlik ve yükseklik kullanarak resmi yükseltelim
up_width = 600
up_height = 400
up_points = (up_width, up_height)
resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)

 
# Resimleri görüntüleyin
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.waitKey()
cv2.imshow('Resized Up image by defining height and width', resized_up)
cv2.waitKey()


# <Ölçekleme Faktörüyle Yeniden Boyutlandırma>
'''
Ölçekleme Faktörü veya Ölçek Faktörü, genellikle bir miktarı ölçekleyen veya çoğaltan bir sayıdır
En boy oranının bozulmadan kalmasına yardımcı olur ve görüntü kalitesini korur. 
Böylece, görüntüyü büyütürken veya küçültürken görüntü bozuk görünmez.
'''

# Her iki ölçekleme faktörünü de belirterek görüntüyü 1,2 kat büyütme
scale_up_x = 1.2
scale_up_y = 1.2

# Tek bir ölçek faktörü belirterek görüntüyü 0,6 kat küçültme.
scale_down = 0.6

scaled_f_down = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
scaled_f_up = cv2.resize(image, None, fx= scale_up_x, fy= scale_up_y, interpolation= cv2.INTER_LINEAR)


cv2.imshow('Resized Down by defining scaling factor', scaled_f_down)
cv2.waitKey()
cv2.imshow('Resized Up image by defining scaling factor', scaled_f_up)
cv2.waitKey()


# <Farklı İnterpolasyon Yöntemleriyle Yeniden Boyutlandırma>
'''
Farklı yeniden boyutlandırma amaçları için farklı enterpolasyon yöntemleri kullanılır.

- INTER_AREA: INTER_AREA yeniden örnekleme için piksel alanı ilişkisini kullanır. 
Bu, bir görüntünün boyutunu küçültmek (küçülmek) için en uygun yöntemdir. 
Görüntüyü yakınlaştırmak için kullanıldığında INTER_NEAREST yöntemi kullanır.

- INTER_CUBIC: Bu, görüntüyü yeniden boyutlandırmak için bikübik enterpolasyon kullanır. 
Bu yöntem, yeni pikselleri yeniden boyutlandırırken ve enterpolasyon yaparken görüntünün komşu 4x4 piksellerine etki eder. 
Ardından, yeni enterpolasyonlu pikseli oluşturmak için 16 pikselin ağırlık ortalamasını alır.

- INTER_LINEAR: Bu yöntem, INTER_CUBIC enterpolasyona biraz benzer. 
Ancak aksine INTER_CUBIC, bu, enterpolasyonlu piksel için ağırlıklı ortalamayı elde etmek için 2x2 komşu piksel kullanır.

- INTER_NEAREST: Bu yöntem, enterpolasyon için en yakın komşu kavramını kullanır. 
Bu, enterpolasyon için görüntüden yalnızca bir komşu piksel kullanan en basit yöntemlerden biridir.
'''

# Farklı İnterpolasyon Yöntemi kullanarak görüntüyü 0,6 kez küçültme

res_inter_nearest = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_NEAREST)
res_inter_linear = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
res_inter_area = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_AREA)

# Karşılaştırma için görüntüleri yatay eksende birleştirin
vertical= np.concatenate((res_inter_nearest, res_inter_linear, res_inter_area), axis = 0)

# Resmi görüntüleyin Devam etmek için herhangi bir tuşa basın
cv2.imshow('Inter Nearest :: Inter Linear :: Inter Area', vertical)
cv2.waitKey()

cv2.destroyAllWindows()