'''
https://learnopencv.com/background-subtraction-with-opencv-and-bgs-libraries/

!!!!!!!!! Kütüphane Hatası Aldın !!!!!!!!!!!!!!!!
'''
import argparse
import cv2
import pybgs as bgs


def get_bgslib_result(video_to_process):
    # video işleme için VideoCapture nesnesi oluşturun

    captured_video = cv2.VideoCapture(video_to_process)
    # video capture durumunu kontrol et
    if not captured_video.isOpened:
        print("Unable to open: " + video_to_process)
        exit(0)

    #  örnek arka plan çıkarma
    background_subtr_method = bgs.SuBSENSE()

    while True:
        # video karelerini oku
        retval, frame = captured_video.read()

        # frame lerin tutulup tutulmadığını kontrol edin
        if not retval:
            break

        # video karelerini yeniden boyutlandır
        frame = cv2.resize(frame, (640, 360))

        # frame leri background subtractor a iletin
        foreground_mask = background_subtr_method.apply(frame)
        # ön plan maskesi olmadan arka planı elde edin
        img_bgmodel = background_subtr_method.getBackgroundModel()

        # geçerli frame i, ön plan maskesini, çıkarılan sonucu göster
        cv2.imshow("Initial Frames", frame)
        cv2.imshow("Foreground Masks", foreground_mask)
        cv2.imshow("Subtraction result", img_bgmodel)

        keyboard = cv2.waitKey(10)
        if keyboard == 27:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video",
        type=str,
        help="Define the full input video path",
        default="space_traffic.mp4",
    )

    # argümentler
    args = parser.parse_args()

    # BS-pipeline başlatın
    get_bgslib_result(args.input_video)


