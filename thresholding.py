'''
https://learnopencv.com/opencv-threshold-python-cpp/

'''
import cv2

src = cv2.imread("_resources/_photos/input_image.jpg", cv2.IMREAD_GRAYSCALE)

# Basic threhold example
'''
threshold(src,dst, thresh, maxValue, THRESH_BINARY);

giriş görüntüsüne ikili threshold uygulamasının sonucunu gösterir. 
thresh=0 ile maxValue = 255 arasındaki değerler gösterilir.
'''
th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY)
cv2.imwrite("_outputs/opencv-threshold-example.jpg", dst)
cv2.imshow("Original", src)
cv2.imshow("threshold-example", dst)
cv2.waitKey()


#  maxValue ile Thresholding 128 olarak ayarlandı
'''
maxValue 128, eşik bölgelerin değerini 128'e ayarlar.
'''
th1, dst1 = cv2.threshold(src, 0, 128, cv2.THRESH_BINARY)
cv2.imshow("Original", src)
cv2.imshow("thresh-binary-maxval.jpg", dst1)
cv2.imwrite("_outputs/opencv-thresh-binary-maxval.jpg", dst1)
cv2.waitKey()
 

# thresh 127 olarak ayarlandı
'''
thresh i 127'ye değiştirmek, 127'ye eşit veya daha küçük tüm sayıları kaldırır.
'''
th2, dst2 = cv2.threshold(src,127,255, cv2.THRESH_BINARY)
cv2.imshow("Original", src)
cv2.imshow("thresh-binary.jpg", dst2)
cv2.imwrite("_outputs/opencv-thresh-binary.jpg", dst2)
cv2.waitKey()

 
# THRESH_BINARY_INV kullanarak  Thresholding oluşturma
'''
!?!?!
'''
th3, dst3 = cv2.threshold(src,127,255, cv2.THRESH_BINARY_INV)
cv2.imshow("Original", src)
cv2.imshow("thresh-binary-inv.jpg", dst3)
cv2.imwrite("_outputs/opencv-thresh-binary-inv.jpg", dst3)
cv2.waitKey()

 
# Thresholding using THRESH_TRUNC
'''
Kaynak piksel değeri eşikten büyükse, hedef piksel eşiğe(thresh) ayarlanır. 
Aksi takdirde kaynak piksel değerine ayarlanır.
maxValue yok sayılıyor 
'''
th4, dst4 = cv2.threshold(src,127,255, cv2.THRESH_TRUNC)
cv2.imshow("Original", src)
cv2.imshow("thresh-trunc.jpg", dst4)
cv2.imwrite("_outputs/opencv-thresh-trunc.jpg", dst4)
cv2.waitKey()

 
# Thresholding using THRESH_TOZERO
'''
Kaynak piksel değeri eşikten büyükse, hedef piksel değeri ilgili kaynağın piksel değerine ayarlanır.
Aksi takdirde, sıfıra ayarlanır
maxValue yok sayılıyor 
'''
th5, dst5 = cv2.threshold(src,127,255, cv2.THRESH_TOZERO)
cv2.imshow("Original", src)
cv2.imshow("thresh-tozero.jpg", dst5)
cv2.imwrite("_outputs/opencv-thresh-tozero.jpg", dst5)
cv2.waitKey()

 
# Thresholding using THRESH_TOZERO_INV
'''
Kaynak piksel değeri eşikten büyükse, hedef piksel değeri sıfıra ayarlanır.
Aksi takdirde kaynak piksel değerine ayarlanır.
maxValue yok sayılıyor  
'''
th6, dst6 = cv2.threshold(src,127,255, cv2.THRESH_TOZERO_INV)
cv2.imshow("Original", src)
cv2.imshow("thresh-to-zero-inv.jpg", dst6)
cv2.imwrite("_outputs/opencv-thresh-to-zero-inv.jpg", dst6)


cv2.waitKey()
cv2.destroyAllWindows()