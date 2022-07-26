import cv2

# cv2.imread() işlevi bir görüntüyü okumak için kullanılır.
'''
cv2.IMREAD_UNCHANGED  veya -1
cv2.IMREAD_GRAYSCALE  veya 0
cv2.IMREAD_COLOR  veya 1 
'''

img_color = cv2.imread('_resources/_images_must/cat.jpg',cv2.IMREAD_COLOR)
img_grayscale = cv2.imread('_resources/_images_must/cat.jpg',cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread('_resources/_images_must/cat.jpg',cv2.IMREAD_UNCHANGED)

# Orijinal yüksekliği ve genişliği alın
'''
image.shape() Python'da üç değer döndürür: Yükseklik(Height), genişlik(width) ve kanal sayısı.
'''
h,w,c = img_color.shape
print("Original Height and Width:", h,"x", w)


# cv2.imshow() işlevi, bir görüntüyü bir pencerede görüntülemek için kullanılır.
cv2.imshow('color image', img_color)
cv2.imshow('graycsale image', img_grayscale)
cv2.imshow('unchanged image', img_unchanged)


# waitKey() pencereyi kapatmak için bir tuşa basılmasını bekler ve ms cinsinden değer alır(0 sonsuz süre)
cv2.waitKey(0)
 
# cv2.destroyAllWindows() oluşturduğumuz tüm pencereleri yok eder.
cv2.destroyAllWindows()
 
# cv2.imwrite() işlevi bir görüntüyü diske yazmak için kullanılır.
cv2.imwrite('_resources/_outputs/cat_grayscale.jpg',img_grayscale)