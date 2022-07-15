'''
Kontur algılamayı kullanarak nesnelerin sınırlarını algılayabilir ve bunları bir görüntüde kolayca konumlandırabiliriz.
Görüntü-ön plan çıkarma, basit görüntü bölütleme, algılama ve tanıma gibi birçok ilginç uygulama için genellikle ilk adımdır. 


https://learnopencv.com/contour-detection-using-opencv-python-c/

'''

import cv2

image = cv2.imread('_resources/_images_must/contour.jpg')


# görüntüyü gray-scale e dönüştürün
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# thresholding uygula
'''
Şimdi, görüntüye threshold() uygulamak için işlevi kullanın. 143'den büyük bir değere sahip herhangi bir piksel 255 (beyaz) 
değerine ayarlanacaktır. Ortaya çıkan görüntüde kalan tüm pikseller 0 (siyah) olarak ayarlanacaktır. 150 eşik değeri 
ayarlanabilir bir parametredir, bu nedenle deneme yapabilirsiniz.
'''
ret, thresh = cv2.threshold(img_gray, 143, 255, cv2.THRESH_BINARY)
# visualize the binary image
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('_outputs/image_thres1.jpg', thresh)
cv2.destroyAllWindows()


# findContours() kullanarak threshold görüntüdeki konturları tespit edin
'''
findContours(image, mode, method, contours=, hierarchy=, offset=)

- image: Girdi görüntü. (Threshold uygulanmış görüntü)
- mode: Bu, kontur alma modudur. Bunu olarak RETR_TREE sağladık, bu da algoritmanın threshold görüntüden tüm olası konturları 
alacağı anlamına gelir. 
- method: Bu, kontur yaklaşımı yöntemini tanımlar. Bu örnekte CHAIN_APPROX_SIMPLE, CHAIN_APPROX_NONE'den biraz daha yavaş olsa da,
burada tüm kontur noktalarını saklamak için bu yöntemi kullanacağız. 
'''
contours,hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


# orijinal görüntüye drawContours() kullanarak konturlar çizin
'''
- image: Bu, konturu çizmek istediğiniz giriş RGB görüntüsüdür.
- contours: findContours() Fonksiyondan elde edilen konturları gösterir.
- contourIdx: Kontur noktalarının piksel koordinatları, elde edilen konturlarda listelenir. Bu argümanı kullanarak, tam olarak
hangi kontur noktasını çizmek istediğinizi belirterek bu listeden indeks konumunu belirleyebilirsiniz. Negatif bir değer 
verilmesi tüm kontur noktalarını çizecektir.
- color: Bu, çizmek istediğiniz kontur noktalarının rengini belirtir. Noktaları yeşille çiziyoruz.
- thickness: Bu, kontur noktalarının kalınlığıdır.
'''
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                

# sonuçları görüntüleyin
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('_outputs/contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()