import glob
import os
import shutil
from IPython.display import display, clear_output, HTML
from core.dataset import Dataset

import numpy as np
import cv2
import pandas as pd

from enum import Enum

class ObjectTypes(Enum):
    Airplane = 0
    Helicopter = 1
    Bird = 2
    Drone = 3
    Flock = 4
    Airborne = 5


class Main():

    def __init__(self):

        anapath='D:/orhan/Belgeler/Datasets/airborne_object_tracking'

        # part1 ve images path yolu
        self.part1_path = anapath+'/airborne-detection-starter-kit/data/part1/'
        self.images_path = self.part1_path+'Images'

        # Klasördeki klasörlerin listesi
        self.folders = [f for f in os.listdir(self.images_path) if os.path.isdir(os.path.join(self.images_path, f))]

        #aws deki url ve yer
        self.dataset = Dataset(self.part1_path, 's3://airborne-obj-detection-challenge-training/part1/', partial=True, prefix='part1')

        #yolo pathleri
        self.yolo_output_dir = os.path.join(anapath, "yolo_dataset/" )

        #dnn pathleri
        self.dnn_output_dir = os.path.join(anapath, "dnn_dataset/" )

        #siamese pathleri
        self.siamese_output_dir = os.path.join(anapath, "siamese_dataset/" )

        #tüm operasyonu ana dataframei
        self.path_ana_islemis_dataframe = anapath+'/ana_islemis_dataframe.csv'
        try:
            self.ana_islemis_dataframe =pd.read_csv(self.path_ana_islemis_dataframe)
        except:
            df_cols = ['flight_id', 'toplam_frame' , 'background_sayisi','all_objects']
            #ObjectTypes.Airplane.name,ObjectTypes.Helicopter.name,ObjectTypes.Bird.name, ObjectTypes.Drone.name,ObjectTypes.Flock.name,ObjectTypes.Airborne.name]
            self.ana_islemis_dataframe = pd.DataFrame(columns=df_cols)

        
        self.height, self.width = 2048 , 2448
        self.fps = 10

        self.kac_kere_calissin = 1


    def run(self):

        kactayiz= 0
        for lucky_flight_folder in self.folders:
            
            self.lucky_flight_id = lucky_flight_folder

            #dataframe daha once varsa pas geç 
            flight_id_var_mi = self.ana_islemis_dataframe['flight_id'].str.contains(self.lucky_flight_id).any()
            if flight_id_var_mi ==True:
                print("flight_id_var: ",self.lucky_flight_id)
                continue

            self.lucky_flight = None
            try:
                self.lucky_flight = self.dataset.get_flight_by_id(self.lucky_flight_id)
            except:
                print("ucus bulunamadi: ",self.lucky_flight_id)
                continue

            kactayiz = kactayiz+1

            if kactayiz>self.kac_kere_calissin:
                print("break durdurdu: ",kactayiz)
                break

            print("üzerine çalışılıyor: ",self.lucky_flight_id)

            '''self.mdPrint("List of Airborne Objects: ")
            for airborne_obj in self.lucky_flight.get_airborne_objects():
                self.mdPrint("- %s " % airborne_obj)'''

            all_keys = []
            all_keys_not_remove= []

            all_keys.extend([self.removeNumbers(k) for k in self.lucky_flight.detected_objects])
            all_keys_not_remove.extend([k for k in self.lucky_flight.detected_objects])

            #unique_keys = list(set(all_keys))
            #print('unique_keys',unique_keys)
            #print('unique_keys_not_remove',str(set(all_keys_not_remove)))

            rows_for_dnn = self.yoloTxtPreSiamese()

            toplam_frame, background_sayisi= self.yoloDatasetPreVideo()

            #dnn
            self.dnnDataset(rows_for_dnn)

            #dataframe kaydet

            self.ana_islemis_dataframe = self.ana_islemis_dataframe.append({'flight_id': str(self.lucky_flight_id),
                                                                        'toplam_frame': str(toplam_frame),
                                                                        'background_sayisi': str(background_sayisi),
                                                                        'all_objects': str(str(set(all_keys_not_remove)))}, 
                                                                        ignore_index=True, verify_integrity=False,
                                                                                sort=False)


        
            self.ana_islemis_dataframe.to_csv(self.path_ana_islemis_dataframe, index=False)

            

    # keys
    def removeNumbers(self,s):
        return ''.join([i for i in s if not i.isdigit()])

    def mdPrint(self, text):
        display({
            'text/markdown': text,
            'text/plain': text
        }, raw=True)

    def dnnDataset(self,rows):

        if os.path.exists(self.dnn_output_dir)==False:
            print('created')
            os.mkdir(self.dnn_output_dir)

        df = pd.DataFrame(rows)
        df.columns = ['flight_id', 'object_type', 'object', 'frame_id', 
                    'left', 'top', 'width', 'height', 'area', 'image_path','range_distance']
        #print(df.head())
        df.to_csv(self.dnn_output_dir+str(self.lucky_flight_id)+".csv", index=False)


    def siameseDataset(self,object_type,obj_key,img_crop,temp_image_path_witout_ext):
        
        if os.path.exists(self.siamese_output_dir)==False:
            print('created')
            os.mkdir(self.siamese_output_dir)

        temp_object_type_folder =self.siamese_output_dir+"/"+object_type
        if os.path.exists(temp_object_type_folder)==False:
            print('created')
            os.mkdir(temp_object_type_folder)

        '''temp_flight_id_folder =self.siamese_output_dir+"/"+object_type
        if os.path.exists(temp_flight_id_folder)==False:
            print('created')
            os.mkdir(temp_flight_id_folder)'''

        temp_image_path = temp_object_type_folder+"/"+temp_image_path_witout_ext+"_"+str(obj_key)+".png"
        cv2.imwrite(temp_image_path,img_crop)
        

    def yoloDatasetPreVideo(self,):

        if os.path.exists(self.yolo_output_dir)==False:
            print('created')
            os.mkdir(self.yolo_output_dir)
        
        current_dir = self.part1_path+'/Images/'+self.lucky_flight_id

        video_out_path = current_dir+"/"+self.lucky_flight_id+'.mp4'
        if os.path.exists(video_out_path):
            vision_frame_save_out = None
        else:
            vision_frame_save_out = cv2.VideoWriter(video_out_path ,cv2.VideoWriter_fourcc(*'xvid'), self.fps, (self.width,self.height)) 

        # bunlar resımler için
        train_yolo_output_dir = os.path.join(self.yolo_output_dir, "train"+"_"+str(self.lucky_flight_id))
        if os.path.exists(train_yolo_output_dir)==False:
            os.mkdir(train_yolo_output_dir)

        train_images_yolo_output_dir = os.path.join(train_yolo_output_dir, "images")
        if os.path.exists(train_images_yolo_output_dir)==False:
            os.mkdir(train_images_yolo_output_dir)

        train_labels_yolo_output_dir = os.path.join(train_yolo_output_dir, "labels")
        if os.path.exists(train_labels_yolo_output_dir)==False:
            os.mkdir(train_labels_yolo_output_dir)

        val_yolo_output_dir = os.path.join(self.yolo_output_dir, "val"+"_"+str(self.lucky_flight_id))
        if os.path.exists(val_yolo_output_dir)==False:
            os.mkdir(val_yolo_output_dir)

        val_images_yolo_output_dir = os.path.join(val_yolo_output_dir, "images")
        if os.path.exists(val_images_yolo_output_dir)==False:
            os.mkdir(val_images_yolo_output_dir)

        #bunlar etıketler ıcın
        val_labels_yolo_output_dir = os.path.join(val_yolo_output_dir, "labels")
        if os.path.exists(val_labels_yolo_output_dir)==False:
            os.mkdir(val_labels_yolo_output_dir)

        split_max_class_count = len(ObjectTypes)

        temp_path_custom_yaml_file = self.yolo_output_dir+'custom.yaml'
        if os.path.exists(temp_path_custom_yaml_file)==False:
            
            file_yaml = open(temp_path_custom_yaml_file, 'w')

            file_yaml.write("path: "+str(self.yolo_output_dir)+ "\n" + "\n")

            file_yaml.write("train:  "+"\n")
            file_yaml.write("   "+"- train"+"_"+str(self.lucky_flight_id) + "\n")
            file_yaml.write("val: "+ "\n")
            file_yaml.write("   "+"- val"+"_"+str(self.lucky_flight_id) + "\n")

            file_yaml.write("\n")
            file_yaml.write("\n")
            file_yaml.write("# number of classes"+"\n")
            file_yaml.write("nc: "+str(split_max_class_count))
            file_yaml.write("\n")
            file_yaml.write("\n")
            file_yaml.write("# class names"+ "\n")

            class_name = "names: ["
            for i in range(int(split_max_class_count)):
                if i == split_max_class_count-1:
                    class_name = class_name+"'"+str(i)+"'"
                else:
                    class_name = class_name+"'"+str(i)+"',"
            class_name = class_name +"]"

            file_yaml.write(class_name+"\n")
            file_yaml.close()

        else:
            read_yaml = open(temp_path_custom_yaml_file, 'r').readlines()

            with open(temp_path_custom_yaml_file, 'w') as outfile:
                for index, line in enumerate(read_yaml):
                    try:
                        #index = int(line.split(" ")[0])
                        if line.strip() == str("train:  "+"\n").strip():
                            outfile.write(line)
                            outfile.write("   "+"- train"+"_"+str(self.lucky_flight_id) + "\n")
                        elif line.strip() == str("val:  "+"\n").strip():
                            outfile.write(line)
                            outfile.write("   "+"- val"+"_"+str(self.lucky_flight_id) + "\n")
                        else:
                            outfile.write(line)
                    except:
                        pass
    
                outfile.close()


        split_percentage_valid = 20

        counter = 1
        background_sayisi=0
        toplam_frame=0
        index_test = round(100 / split_percentage_valid)
        black_img = np.zeros((self.height,self.width,3), dtype = np.uint8)
        for png_file in glob.iglob(os.path.join(current_dir, '*.png')):
            title, ext = os.path.splitext(os.path.basename(png_file))
            txt_file = os.path.join(current_dir, title+'.txt')

            try:
                frame = cv2.imread(png_file)
                if vision_frame_save_out is not None:
                    vision_frame_save_out.write(frame)
                    print("-------video olusturuluyor-----")
                else:
                    print("-------video pas geciyor-----")
            except:
                print("try-catch save_vision_frame_save")
                pass
            
            if counter == index_test:
                counter = 1
                try:
                    shutil.move(png_file, val_images_yolo_output_dir)
                    shutil.move(txt_file, val_labels_yolo_output_dir)
                except:
                    #print("txt yok olabılır"+ext,title,ext)
                    background_sayisi = background_sayisi +1
            else:
                try:
                    shutil.move(png_file, train_images_yolo_output_dir)
                    shutil.move(txt_file, train_labels_yolo_output_dir)
                except:
                    #print("txt yok olabılır"+ext,title,ext)
                    background_sayisi = background_sayisi +1
                counter = counter + 1




            toplam_frame = toplam_frame + 1
            cv2.imwrite(png_file, black_img)


        print('background_sayisi_sayisi',background_sayisi)
        return toplam_frame, background_sayisi
            
    def yoloTxtPreSiamese(self,):

        # oncekı txtleri siler
        print_show = False
        tifCounter = 0
        myPath = self.part1_path+'/Images/'+self.lucky_flight_id

        if print_show:
            print(myPath)
        for root, dirs, files in os.walk(myPath):
            for file in files:    
                if file.endswith('.txt'):
                    #print(file)
                    os.remove(myPath+'/'+file)
                    tifCounter += 1

        rows = []
        

        
        resim_indir = 0
        for obj_key in self.lucky_flight.detected_objects:
            object_type = self.removeNumbers(obj_key)
            if print_show:
                print('object_type',object_type)
            obj = self.lucky_flight.detected_objects[obj_key]

            if print_show:
                print('obj',obj)
            
            for loc in obj.location:

                resim_indir = resim_indir + 1
                bbox = loc.bb.get_bbox()
                #print('bbox',bbox)
                frame_id = loc.frame.id
                #print('frame_id',frame_id)
                range_distance = loc.range_distance_m
                #print('range_distance',range_distance)
                image_path = loc.frame.image_path()
                image_base_name=os.path.basename(loc.frame.image_path())
                image_path = myPath + "/"+image_base_name
                #print('image_path',image_path)
                rows.append([self.lucky_flight_id, object_type, obj_key, frame_id,*bbox, bbox[-1]*bbox[-2], image_path, range_distance])

                (x, y, w, h) = [int(v) for v in bbox]

                center_x = (x + (x+w))/2
                center_y = (y +(y+h))/2
                yolo_x = format(center_x/self.width, '.6f')
                yolo_y = format(center_y/self.height, '.6f')

                yolo_w = format(w/self.width, '.6f')
                yolo_h = format(h/self.height, '.6f')

                enum_value = ObjectTypes[object_type].value
                yolo_line = '{0} {1} {2} {3} {4}'.format(enum_value, yolo_x, yolo_y, yolo_w, yolo_h)


                temp_image_path_witout_ext = image_base_name.split(".")[0]
                txt_path= image_path.split(".")[0]

                if print_show:
                    print('image_path:', image_path)
                    print('1: ',txt_path)
                txt_path=txt_path+".txt"
                if os.path.exists(txt_path)==False:
                    file = open(txt_path, 'w')
                    file.close()


                infile = open(txt_path,'r', encoding='utf-8').readlines()
                with open(txt_path, 'w', encoding='utf-8') as outfile:
                    outfile.writelines(infile)
                    outfile.writelines(yolo_line+"\n")


                img_crop = cv2.imread(image_path)
                img_crop = img_crop[y:y+h,x:x+w]
                self.siameseDataset(object_type,obj_key,img_crop,temp_image_path_witout_ext)
                

                if print_show:
                    print('--------------------------')



        return rows





if __name__ == '__main__':
    main = Main()
    main.run()
