'''
Translation

Bilgisayarla görmede, bir görüntünün çevrilmesi, onu x ve y eksenleri boyunca belirli 
sayıda piksel kaydırmak anlamına gelir. Görüntünün kaydırılması için gereken pikseller tx ve ty olsun.

https://learnopencv.com/image-rotation-and-translation-using-opencv/

'''
import cv2
import numpy as np

image = cv2.imread('_resources/_photos/cat.jpg')

# görüntünün genişliğini ve yüksekliğini alın
height, width = image.shape[:2]

# translation için tx ve ty değerlerini alın
# istediğiniz herhangi bir değeri belirtebilirsiniz
tx, ty = width / 4, height / 4
 
# tx ve ty kullanarak translation matrisini oluşturun, bu bir NumPy dizisidir
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)


# resme translation uygula
'''
- warpAffine() bir görüntüye herhangi bir tür affine dönüştürme uygulamak için kullanılabilen genel bir işlevdir. 
Sadece M matrisini uygun şekilde tanımlayın.
'''
translated_image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))

# orijinal ve çevrilmiş görüntüleri göster
cv2.imshow('Translated image', translated_image)
cv2.imshow('Original image', image)
cv2.waitKey(0)

# # çevirilen görüntüyü diske kaydet
cv2.imwrite('_outputs/translated_cat_image.jpg', translated_image)