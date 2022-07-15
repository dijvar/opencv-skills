'''
https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

!!!!!!!!!!!! HATA ALINDI COZUM BULUNAMADI !!!!!!!!!!!!!!!!
'''
import cv2 as cv
import math
import argparse

parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# Input argument
parser.add_argument('--input', default= 'multiple_plates.jpg', 
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.'
                    )
# Model argument
parser.add_argument('--model', default="frozen_east_text_detection.pb",
                    help='Path to a binary .pb file of model contains trained weights.'
                    )
# Width argument
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
                   )
# Height argument
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
                   )
# Confidence threshold
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.'
                   )
# Non-maximum suppression threshold
parser.add_argument('--nms',type=float, default=0.4,
                    help='Non-maximum suppression threshold.'
                   )

args = parser.parse_args()


############ Utility fonksiyonları ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ Geometri ve puanların BOYUTLARINI VE ŞEKİLLERİNİ KONTROL EDİN ############
    assert len(scores.shape) == 4, "Skorların yanlış boyutları"
    assert len(geometry.shape) == 4, "Yanlış geometri boyutları"
    assert scores.shape[0] == 1, "Geçersiz puan boyutları"
    assert geometry.shape[0] == 1, "Geçersiz geometri boyutları"
    assert scores.shape[1] == 1, "Geçersiz puan boyutları"
    assert geometry.shape[1] == 5, "Geçersiz geometri boyutları"
    assert scores.shape[2] == geometry.shape[2], "Puanların ve geometrinin geçersiz boyutları"
    assert scores.shape[3] == geometry.shape[3], "Puanların ve geometrinin geçersiz boyutları"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Puanlardan veri ayıklayın
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # Puan eşik puanından düşükse sonraki x'e geçin
            if(score < scoreThresh):
                continue

            # Offset i hesapla
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Açının cos ve sin i hesaplayın
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Offset i hesapla
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Dikdörtgen için puan bul
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # detections ve confidences return et
    return [detections, confidences]

if __name__ == "__main__":
    # Argümanları okuyun ve saklayın
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model

    # Network(Ağı) ü yükle
    net = cv.dnn.readNet(model)

    # Yeni bir adlandırılmış pencere oluştur
    kWinName = "EAST: An Efficient and Accurate Scene Text Detector(Verimli ve Dogru Bir Metin Dedektoru)"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    # Bir video dosyası veya bir görüntü dosyası veya bir kamera akışı açın
    cap = cv.VideoCapture(args.input if args.input else 0)

    while cv.waitKey(1) < 0:
        # Frame leri oku
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        # Frame lerin width ve height larını al
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Frame den bir 4B blob oluşturun.
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Modeli çalıştırın
        net.setInput(blob)
        output = net.forward(outputLayers)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        # Puanları ve geometriyi alın
        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)
        # NMS ekleyin
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        for i in indices:
            # döndürülmüş doğrunun 4 köşesini alın
            vertices = cv.boxPoints(boxes[i[0]])
            # ilgili oranlara göre sınırlayıcı kutu koordinatlarını ölçeklendir
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                cv.line(frame, p1, p2, (0, 255, 0), 2, cv.LINE_AA)
                # cv.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

        # Verimlilik bilgilerini koy
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Frame göster
        cv.imshow(kWinName,frame)
        cv.imwrite("_outputs/out-{}".format(args.input),frame)
