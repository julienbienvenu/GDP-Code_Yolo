# Class to store all the info of the xml file
import os
import random
import numpy as np
import shutil
import cv2
import sys
import pandas as pd
import glob
import torch
from keras.models import load_model

from PoseEstimation.pipe_detection import PipeDetection
from fcnn import add_line_to_file, get_top3_values_and_indexes
from interface import Interface

# Load model
CLASSIFIER = load_model('models/RNN.h5')
PIPEDETECTION = PipeDetection()

class Frame_light():

    def __init__(self, x1=0, x2=0, y1=0, y2=0, 
            fileroot = 'None', marshall_signal = 0,
            frame = ""
            ):

        #Infos
        self.fileroot = fileroot
        self.marshall_signal = marshall_signal #Class
        self.frame = frame

        #Bounding box
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2 

    # Resize the frame for gesture detection
    def resize_frame(self, frame = 'None'):
        if frame != 'None':
            return self.frame[int(self.y1) : int(self.y2+1), int(self.x1+1) : int(self.x2+1)]
        else:
            # frame = cv2.imread(self.fileroot)
            self.frame = self.frame[int(self.y1) : int(self.y2+1), int(self.x1+1) : int(self.x2+1)]
        
    def detect(self):
        # Detect the marshall and get the bounding box
        # Based on detect.py from Yolo
        # - conf : minimun confidence
        # - weighted model
        # - image path

        with torch.no_grad():
            
            sys.path.append('yolov7/')
            from detect_marshall import detect_marshall
            values = detect_marshall(frame = self.frame)
            print(values)

        if len(values) == []:
            print("No marshall found")
        
        else :

            img_height, img_width, _ = self.frame.shape

            self.x1 = (values[0]-(values[2]/2))*img_width
            self.y1 = (values[1]-(values[3]/2))*img_height
            self.x2 = (values[0]+(values[2]/2))*img_width
            self.y2 = (values[1]+(values[3]/2))*img_height   
            print(f"{self.x1}:{self.y1}:{self.x2}:{self.y2}")

    def plot(self):

        img = cv2.imread(self.fileroot)
        # Draw the bounding box on the image
        cv2.rectangle(img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), (0, 255, 0), 2)

        # Show the image with the bounding box
        cv2.imshow('Image with bounding box', img)

        # Wait for a key press and then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# This class store all the Frame parameters of one video
# Not USE
class Video_light():

    def __init__(self, filename = 'None', interface = Interface()):
        
        self.filename = filename
        self.bbox_list = pd.DataFrame({'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []})
        
        # Add interface
        self.interface = interface

        # Store the video to a list of frames
        self.frames = []
        self.frame_count = 0     

    def detection(self, yolo = False, loading_data = False):

        if yolo :

            # Generate all the bbox
            for frame in self.frames:
                if frame.fileroot not in self.bbox_list.index:
                    frame.detect()
                    new_row = pd.DataFrame({'xmin': frame.x1, 'xmax': frame.x2, 'ymin': frame.y1, 'ymax': frame.y2}, index=[frame.fileroot])
                    self.bbox_list.loc[frame.fileroot] = new_row.loc[frame.fileroot]

            self.x1 = min(self.bbox_list['xmin'])
            self.x2 = max(self.bbox_list['xmax'])
            self.y1 = min(self.bbox_list['ymin'])
            self.y2 = max(self.bbox_list['ymax'])

        else :

            self.x1, self.x2, self.y1, self.y2 = 200, 400, 50, 400

        if not loading_data:

            # Resize the video to this shape and detect the posture
            self.posture()

            # Classify the movement
            self.classify()

            # Kill the thread
            return

    def update(self, frame = None, ite = 0):

        if isinstance(frame, list):
            nb_frames = len(frame)
        else :
            nb_frames = 1

        if len(self.frames) >= 14:

            self.frames = self.frames[nb_frames:]
            self.frames.append(Frame_light(frame = frame)) 

        else :

            frame_name = f"{self.interface.eventID}_{ite}"
            # cv2.imwrite(frame_name, frame)
            self.frames.append(Frame_light(frame = frame, fileroot = frame_name))        

    def classify(self, list_input = None):

        if list_input is not None:
            self.video_angles = list_input

        if len(self.video_angles) == 14:            

            # Predict with the model
            self.video_angles = np.reshape(self.video_angles, (1, 14, 4))
            output = CLASSIFIER.predict(self.video_angles, verbose = 0)
            value = np.argmax(output, axis=1)                

            print(f"Predicted class : {value+1} \n {output}")

            # add_line_to_file(value, output[value])

            # Generate Kafka exception
            # self.interface.json_output(value)


    def posture(self, write_files_norm = False, write_files_angles = False):

        # List of values
        self.video_angles = []
        # self.video_norm = []

        for frame in self.frames:

            frame.x1 = self.x1
            frame.x2 = self.x2
            frame.y1 = self.y1
            frame.y2 = self.y2

            frame.resize_frame()

            # Call Posture function
            self.video_angles.append(PIPEDETECTION.get_angles(frame.frame))

            if len(self.video_angles) > 14:
                self.video_angles = self.video_angles[-14:]


if __name__ == '__main__':

    video = Video_light()
    print(np.shape([[0,0,0,0] for _ in range(14)]))
    video.classify(list_input=[[0,0,0,0] for _ in range(14)])