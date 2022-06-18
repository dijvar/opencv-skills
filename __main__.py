import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('tools')
from configmanager import ConfigurationManager
from videocapture import VideoCapture

import cv2
import time
import argparse

def main():
    """
    İslemin basladigi ana siniftir.
    """
   

    while True:
        
        video_capture.get_image()

        if (video_capture.ret == False):
            break

        


if __name__ == '__main__':
    '''configurationManager = ConfigurationManager()
    
    parser = argparse.ArgumentParser()
    
    class_id = configurationManager.config_readable['selected_class_id']
    last_frame = configurationManager.config_changeable['last_frame']

    parser.add_argument('-c', '--selected_class_id', help="Secilecek class id", default=class_id)
    parser.add_argument('-l', '--last_frame', help="Baslanilmasi istenen frame numarasi", default=last_frame)
    
    args = parser.parse_args()
    
    
    configurationManager.set_selected_id(selected_id=str(args.selected_class_id))
    configurationManager.set_last_frame(last_frame=str(args.last_frame))'''
    
    
    video_capture = VideoCapture(vision_frame_save=False)

    main()