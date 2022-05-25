'''
https://learnopencv.com/image-rotation-and-translation-using-opencv/

'''
import cv2
import numpy as np


image = cv2.imread('_resources/_photos/cat.jpg')

# görüntünün merkezini elde etmek için yüksekliği ve genişliği 2'ye böl
height, width = image.shape[:2]
# 2B rotation matrisini oluşturmak için görüntünün merkez koordinatlarını alın
center = (width/2, height/2)
 

# rotation matrisini almak için cv2.getRotationMatrix2D() kullanılır
'''
https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html

getRotationMatrix2D(center, angle, scale)

- 2B rotation matrisini oluşturma işlevini sağlar.

center: Giriş görüntüsü için dönme merkezi
angle: Derece cinsinden dönme açısı, pozitif ise saat yönünün tersine döndürülür.
scale: Sağlanan değere göre görüntüyü yukarı veya aşağı ölçekleyen bir izotropik ölçek faktörü
'''
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
 


# cv2.warpAffine() kullanarak görüntüyü döndürün
'''
https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html

warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]])

-warpAffine() görüntüye bir affine dönüşüm uygular. Affine dönüşüm uygulandıktan sonra, 
orijinal görüntüdeki tüm paralel çizgiler çıkış görüntüsünde de paralel kalacaktır.

- warpAffine() bir görüntüye herhangi bir tür affine dönüştürme uygulamak için kullanılabilen genel bir işlevdir. 
Sadece M matrisini uygun şekilde tanımlayın.

src: kaynak büyücü
M: dönüşüm matrisi
dsize: çıktı görüntüsünün boyutu
dst: çıktı görüntüsü
flags: INTER_LINEAR veya INTER_NEAREST gibi enterpolasyon yöntemlerinin kombinasyonu
borderMode(sınır modu): piksel ekstrapolasyon yöntemi, borderMode= BORDER_TRANSPARENT olduğunda, kaynak görüntüdeki "aykırı değerlere" 
karşılık gelen hedef görüntüdeki piksellerin işlev tarafından değiştirilmediği anlamına gelir.
borderValue(sınır değeri): sabit bir sınır olması durumunda kullanılacak değerin varsayılan değeri 0'dır.
'''
rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

cv2.imshow('Original image', image)
cv2.imshow('Rotated image', rotated_image)

cv2.waitKey(0)

# döndürülen görüntüyü diske kaydet
cv2.imwrite('_outputs/rotated_cat_image.jpg', rotated_image)