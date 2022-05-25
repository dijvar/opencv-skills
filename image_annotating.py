'''
https://learnopencv.com/annotating-images-using-opencv/

'''
import cv2

img = cv2.imread('_resources/_photos/sample.jpg')

# Resmi görüntüle
cv2.imshow('Original Image',img)
cv2.waitKey(0)

# Resim boşsa hata mesajı yazdır
if img is None:
    print('Could not read image')

# <Resmin üzerine çizgi çiz>

# copy() kullanarak orijinal görüntünün bir kopyasını oluşturun
imageLine = img.copy()

# line() kullanarak A noktasından B noktasına çizgi çizin
'''
line(image, start_point, end_point, color, thickness, lineType)

OpenCV'deki tüm çizim fonksiyonlarında olduğu gibi, ilk argüman görüntüdür.
Sonraki iki argüman, çizginin başlangıç noktası ve bitiş noktasıdır.
color çizgi rengidir.
thickness kalınlıktır.
lineType çizgi tipidir. İsteğe bağlı argüman.
'''
pointA = (200,80)
pointB = (450,80)
cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
cv2.imshow('Image Line', imageLine)
cv2.waitKey(0)



# <Bir Daire çizin>

# copy() kullanarak orijinal görüntünün bir kopyasını oluşturun
imageCircle = img.copy()
imageFilledCircle = img.copy()

# dairenin merkezini tanımlayın
circle_center = (415,190)

# dairenin yarıçapını tanımlayın
radius =100

# circle() kullanarak bir daire çizin
'''
circle(image, center_coordinates, radius, color, thickness, lineType)

OpenCV'deki tüm çizim fonksiyonlarında olduğu gibi, ilk argüman görüntüdür.
Sonraki iki argüman, dairenin merkezi ve yarıçapı için koordinatları tanımlar.
Sonraki iki argüman, çizginin rengini ve kalınlığını belirtir. !!! thickness -1 seçilirse dolu bir daire çizilir.
lineType çizgi tipidir. İsteğe bağlı argüman.
'''
cv2.circle(imageCircle, circle_center, radius, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
cv2.circle(imageFilledCircle, circle_center, radius, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
cv2.imshow("Image Circle",imageCircle)
cv2.imshow('Image with Filled Circle',imageFilledCircle)
cv2.waitKey(0)



# <Dikdörtgen çizin>

# copy() kullanarak orijinal görüntünün bir kopyasını oluşturun
imageRectangle = img.copy()
imageRectangleFilled = img.copy()

# dikdörtgenin başlangıç ​​ve bitiş noktalarını tanımlayın
start_point =(300,115)
end_point =(475,225)

# rectangle() kullanarak bir dikdörtgen çiz
'''
rectangle(image, start_point, end_point, color, thickness)

OpenCV'deki tüm çizim fonksiyonlarında olduğu gibi, ilk argüman görüntüdür.
Sonraki iki argüman,  dikdörtgenin köşeleri, başlangıç noktası (sol üst) ve bitiş noktası (sağ alt) için 
koordinatları tanımlar.
Sonraki iki argüman, çizginin rengini ve kalınlığını belirtir. !!! thickness -1 seçilirse dolu bir dikdörtgen çizilir.
lineType çizgi tipidir. İsteğe bağlı argüman.
'''
cv2.rectangle(imageRectangle, start_point, end_point, (0, 0, 255), thickness= 3, lineType=cv2.LINE_8)
cv2.rectangle(imageRectangleFilled, start_point, end_point, (0, 0, 255), thickness= -1, lineType=cv2.LINE_8)
cv2.imshow('imageRectangle', imageRectangle)
cv2.imshow('imageFilledRectangle', imageRectangleFilled)
cv2.waitKey(0)



# <Elips Çizin>

# copy() kullanarak orijinal görüntünün bir kopyasını oluşturun
imageEllipse = img.copy()
imageFilledEllipse = img.copy()

# elipsin  merkezini tanımlayın
ellipse_center = (415,190)

# elipsin büyük ve küçük eksenlerini tanımlayın
axis1 = (100,50)
axis2 = (125,50)

# ellipse() kullanarak bir elips çizin
'''
ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color, thickness)
OpenCV'deki tüm çizim fonksiyonlarında olduğu gibi, ilk argüman görüntüdür.
centerCoordinates: elipsin merkez koordinatları
axesLength: elipsin eksen uzunlukları
angle: dönüş açısı
startAngle, endAngle: elipsin başlangıç ve bitiş açısı. Bu açılar ile yayın ne kadarlık bir kısmını 
çizmek istediğimize karar verilir.
Son iki argüman, çizginin rengini ve kalınlığını belirtir. !!! thickness -1 seçilirse dolu bir elips çizilir.
'''
cv2.ellipse(imageEllipse, ellipse_center, axis1, 0, 0, 360, (255, 0, 0), thickness=3) #Horizontal(yatay) mavi elips
cv2.ellipse(imageEllipse, ellipse_center, axis2, 90, 0, 360, (0, 0, 255), thickness=3) #Vertical(dikey) kırmızı elips
cv2.ellipse(imageFilledEllipse, ellipse_center, axis1, 0, 0, 360, (255, 0, 0), thickness=-1) #Horizontal(yatay) dolu mavi elips
cv2.imshow('Ellipse Image',imageEllipse)
cv2.imshow('Filled Ellipse Image',imageFilledEllipse)
cv2.waitKey(0)



# <Yarım Elips Çizin>

# copy() kullanarak orijinal görüntünün bir kopyasını oluşturun
halfEllipse = img.copy()
halfFilledEllipse = img.copy()

# elipsin  merkezini tanımlayın
ellipse_center = (415,190)

# eksen noktasını tanımla
axis1 = (100,50)

# Yarım Elips çizin, sadece anahat
'''
startAngle, elips için 180 derece olarak  ayarlayın
'''
cv2.ellipse(halfEllipse, ellipse_center, axis1, 0, 180, 360, (255, 0, 0), thickness=3)

# Yarım Dolu bir elips çizmek istiyorsanız, bu kod satırını kullanın.
cv2.ellipse(halfFilledEllipse, ellipse_center, axis1, 0, 0, 180, (0, 0, 255), thickness=-1)

cv2.imshow('halfEllipse',halfEllipse)
cv2.imshow('halfFilledEllipse',halfFilledEllipse)
cv2.waitKey(0)



# <Metin Ekleme>

# copy() kullanarak orijinal görüntünün bir kopyasını oluşturun
imageText = img.copy()

# görselin üzerine koymak istediğiniz metni yazalım
text = 'I am a Happy dog!'

# org: Metni nereye koymak istiyorsunuz?
org = (50,350)

# putText() kullanarak metni giriş resmine yaz
'''
putText(image, text, org, font, fontScale, color)

OpenCV'deki tüm çizim fonksiyonlarında olduğu gibi, ilk argüman görüntüdür.
Sonraki argüman, görüntüye açıklama eklemek istediğimiz gerçek metin dizesidir.
Üçüncü argüman, metin dizesinin sol üst köşesi için başlangıç konumunu belirtir. 
Sonraki iki argüman yazı tipi stilini ve ölçeğini belirtir.
color metin rengini belirtir.

OpenCV, Hershey yazı tipi koleksiyonundan birkaç yazı tipi-yüz stilini ve bir italik yazı tipini de destekler. 
Bu listeye göz atın: 
  FONT_HERSHEY_SIMPLEX        = 0,
  FONT_HERSHEY_PLAIN          = 1,
  FONT_HERSHEY_DUPLEX         = 2,
  FONT_HERSHEY_COMPLEX        = 3,
  FONT_HERSHEY_TRIPLEX        = 4,
  FONT_HERSHEY_COMPLEX_SMALL  = 5,
  FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
  FONT_HERSHEY_SCRIPT_COMPLEX = 7,
  FONT_ITALIC                 = 16
'''
cv2.putText(imageText, text, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (250,225,100))


cv2.imshow("Image Text",imageText)

cv2.waitKey(0)
cv2.destroyAllWindows()

