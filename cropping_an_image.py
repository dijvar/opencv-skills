"""

"""
# Import packages
import cv2
import numpy as np

img = cv2.imread('_resources/_photos/cats.jpg')
print(img.shape) # Print image shape
cv2.imshow("original", img)
 

# Bir görüntüyü kırpma
'''
İlk boyut, görüntünün satır sayısı veya yüksekliğidir.
İkinci boyut, görüntünün sütun sayısı veya  genişliğidir. 
cropped = img[start_row:end_row, start_col:end_col]
'''
cropped_image = img[80:280, 0:560]

# Kırpılan parçayı görüntüleme
cv2.imshow("cropped", cropped_image)
 
# Kırpılan parçayı kaydetme
cv2.imwrite("_outputs/cropped_image.jpg", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

# görüntüyü yamalara/parçalara ayırma
img =  cv2.imread("cropped_image.jpg")
image_copy = img.copy()
img_height=img.shape[0]
img_width=img.shape[1]

M = 40  # height
N = 140  # width
x1 = 0
y1 = 0
 
for y in range(0, img_height, M):
    for x in range(0, img_width, N):
        if (img_height - y) < M or (img_width - x) < N:
            break
             
        y1 = y + M
        x1 = x + N
 
        # yama genişliğinin veya yüksekliğinin görüntü genişliğini veya yüksekliğini aşıp aşmadığını kontrol edilir
        if x1 >= img_width and y1 >= img_height:
            x1 = img_width - 1
            y1 = img_height - 1
            #MxN boyutunda yamalar halinde kırpın
            tiles = image_copy[y:y+M, x:x+N]
            #Her yamayı dosya dizinine kaydet
            cv2.imwrite('_outputs/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        elif y1 >= img_height: # yama yüksekliği görüntü yüksekliğini aştığında
            y1 = img_height - 1
            #MxN boyutunda yamalar halinde kırpın
            tiles = image_copy[y:y+M, x:x+N]
            #Her yamayı dosya dizinine kaydet
            cv2.imwrite('_outputs/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        elif x1 >= img_width: # yama genişliği görüntü genişliğini aştığında
            x1 = img_width - 1
            #MxN boyutunda yamalar halinde kırpın
            tiles = image_copy[y:y+M, x:x+N]
            #Her yamayı dosya dizinine kaydet
            cv2.imwrite('_outputs/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)
        else:
            #MxN boyutunda yamalar halinde kırpın
            tiles = image_copy[y:y+M, x:x+N]
            #Her yamayı dosya dizinine kaydet
            cv2.imwrite('_outputs/'+'tile'+str(x)+'_'+str(y)+'.jpg', tiles)
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 1)

#Tam resmi dosya dizinine kaydet
cv2.imshow("patched_image",img)
cv2.imwrite("_outputs/patched.jpg",img)
  
cv2.waitKey()
cv2.destroyAllWindows()
