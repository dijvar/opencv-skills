'''
https://learnopencv.com/blob-detection-using-opencv-python-c/


Blob, bir görüntüdeki bazı ortak özellikleri paylaşan bir grup bağlantılı pikseldir (Ör. grayscale value).

'''

import cv2
import numpy as np

img = cv2.imread("_resources/_photos/blob.jpg", cv2.IMREAD_GRAYSCALE)

# SimpleBlobDetector parametrelerini ayarlayın.
params = cv2.SimpleBlobDetector_Params()

# threshold ları değiştirin
'''
Eşikleme: Kaynak görüntüyü minThreshold'dan başlayan  eşiklerle eşleyerek kaynak görüntüleri 
birkaç ikili görüntüye dönüştürün. Bu eşikler, maxThreshold değerine kadar  eşik Adımı tarafından  artırılır. Dolayısıyla ilk eşik 
minThreshold,  ikincisi  minThreshold  +  eşikAdım ,  üçüncüsü  minThreshold  + 2 x eşikAdım ve bu böyle devam eder.
'''
params.minThreshold = 10
params.maxThreshold = 200



# Alana göre filtreleyin.
'''
Boyuta Göre: filterByArea = True, minArea ve maxArea için değerleri ayarlayarak blobları boyuta göre filtreleyebilirsiniz. 
Örneğin minArea  = 100 ayarı, 100 pikselden daha az olan tüm blobları filtreleyecektir.
'''
params.filterByArea = True
params.minArea = 1500


# Daireselliğe göre filtrele
'''
Bu sadece bloğun bir daireye ne kadar yakın olduğunu ölçer. Bir dairenin daireselliği 1, karenin daireselliği ise 0,785 dir. 
Daireselliğe göre filtrelemek için filterByCircularity = True olarak ayarlayın. Ardından minCircularity  ve maxCircularity
için uygun değerleri ayarlayın.
'''
params.filterByCircularity = True
params.minCircularity = 0.1


# Dışbükeyliğe göre filtrele
'''
Dışbükeylik (Blobun Alanı / dışbükey gövdesinin Alanı) olarak tanımlanır. Dışbükeyliğe göre filtrelemek için
filterByConvexity = True olarak ayarlayın , ardından 0 ≤  minConvexity ≤ 1ve maxConvexity ( ≤ 1)  ayarını yapın
'''
params.filterByConvexity = True
params.minConvexity = 0.87
    

# Eylemsizliğe göre filtrele
'''
Bu bir şeklin ne kadar uzun olduğunu ölçer. Örneğin, daire için bu değer 1'dir, elips için 0 ile 1 arasında ve çizgi için 0'dır.
Atalet oranına göre filtrelemek için  filterByInertia = True olarak  ayarlayın ve (0 ≤ minInertiaRatio ≤ 1)
ve  (maxInertiaRatio ≤ 1 olarak ayarlayın)
'''
params.filterByInertia = True
params.minInertiaRatio = 0.01


# Parametlerle detector oluşturun
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


# Blobları yakalayın
keypoints = detector.detect(img)

# Algılanan blobları kırmızı daireler ile çizerek belirtin.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS bunu sağlar
# dairenin boyutu blobun boyutuna karşılık gelir

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow("Keypoints", im_with_keypoints)
cv2.imwrite("_outputs/blobs_keypoints.jpg", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()