# class to store all the info of the xml file
import os
import random
import numpy as np
import shutil
import cv2
import sys

import torch

class Frame():

    def __init__(self, x1, x2, y1, y2, 
            filename = 'None', fileroot = 'None', marshall_signal = 0,
            directory = 'None', yolo_name = 'None',
            orientation = 'Front', time = 'day',
            img_width = 0, img_height = 0,
            rain = 'False', wind = 'False', snow = 'False', fog = 'False'
            ):

        #Infos
        self.filename = filename
        self.fileroot = fileroot
        self.directory = directory
        self.marshall_signal = marshall_signal #Class
        self.yolo_name = yolo_name #Class Yolo {Aircraft, Marshaller, Workers}

        #Bounding box
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        #Videos parameters
        self.orientation = orientation #{front, back}
        self.rain = rain
        self.wind = wind
        self.snow = snow
        self.fog = fog
        self.time = time #{day,night}

        #Img size
        self.img_width = img_width
        self.img_height = img_height        

    # Resize the frame for gesture detection
    def resize_frame(self, frame):
        return frame[self.x1 : self.x2, self.y1 : self.y2]

    # Show info
    def show(self):
        print(f'Name : {self.filename}')
        print(f'Marshall class : {self.marshall_signal}, Yolo class : {self.yolo_name}')
        print(f'Bounding box : [{self.x1}, {self.y1}, {self.x2}, {self.y2}]')
        print(f'Weather : [Snow : {self.snow}, Fog : {self.fog}, Wind : {self.wind}, Rain : {self.rain}]')

    def yolo_normalization(self):
        self.x1 = round(((self.x1 + self.x2) / 2) / self.img_width,5)
        self.y1 = round(((self.y1 + self.y2) / 2) / self.img_height,5)
        self.x2 = round(((self.x2 + self.x1) / self.img_width) ,5)
        self.y2 = round(((self.y2 + self.y1) / self.img_height),5)

    def write_txt_label(self, marshall_classification = False, signal_classification = False, path_final = 'None'):
        # Randomly separate the labels train/val
        # Write them on the Yolov7 folders from .XML and .PNG files in dataset
        
        path_to_yolo = 'yolov7/data/'+path_final+'/'

        #Write PNG in Yolo
        shutil.copy('dataset\\videos\\'+self.fileroot[15:-4]+'.jpg', path_to_yolo+self.filename)

        #Normalize the label for Yolo
        self.yolo_normalization()

        with open(path_to_yolo+self.filename[:-4]+'.txt', 'a') as f:

            if marshall_classification:

                if self.yolo_name == 'Worker':
                    # print('Write worker')
                    np.savetxt(f, np.column_stack(np.array([1, self.x1, self.y1, self.x2, self.y2])), fmt='%i %f %f %f %f', newline='\n')
                elif self.yolo_name == 'Aircraft':
                    # print('Write Aircraft')
                    np.savetxt(f, np.column_stack(np.array([2, self.x1, self.y1, self.x2, self.y2])), fmt='%i %f %f %f %f', newline='\n')
                elif self.yolo_name == 'Marshaller':
                    # print('Write Marsh')
                    np.savetxt(f, np.column_stack(np.array([0, self.x1, self.y1, self.x2, self.y2])), fmt='%i %f %f %f %f', newline='\n')

            elif signal_classification:
                np.savetxt(f, np.column_stack(np.array([self.marshall_signal, self.x1, self.y1, self.x2, self.y2])), fmt='%i %f %f %f %f', newline='\n')

            f.close()        

    def detect(self, weight = 'yolov7/yolov7.pt', conf = 0.4, image = 'test/image.png'):
        # Detect the marshall and get the bounding box
        # Based on detect.py from Yolo
        # - conf : minimun confidence
        # - weighted model
        # - image path

        with torch.no_grad():
            
            sys.path.append('yolov7/')
            from detect_marshall import detect_marshall
            values = detect_marshall(source = self.fileroot)

        if len(values) == 0:
            print("No marshall found")
        
        else :
            self.x1 = (values[0]-(values[3]/2))*self.img_width
            self.y1 = (values[1]-(values[2]/2))*self.img_height
            self.x2 = (values[0]+(values[3]/2))*self.img_width
            self.y2 = (values[1]+(values[2]/2))*self.img_height   




# This class store all the Frame parameters of one video
# Not USE
class Video():
    def __init__(self, name = 'None', list_frames = []):
        self.name = name
        self.list_frames = list_frames