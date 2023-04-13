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
from interface import Interface

# Load model
CLASSIFIER = load_model('models/Random Forest.h5')

class Frame():

    def __init__(self, x1=0, x2=0, y1=0, y2=0, 
            filename = 'None', fileroot = 'None', marshall_signal = 0,
            directory = 'None', yolo_name = 'None',
            orientation = 'Front', time = 'day',
            img_width = 0, img_height = 0,
            rain = 'False', wind = 'False', snow = 'False', fog = 'False',
            frame = ""
            ):

        #Infos
        self.filename = filename
        # self.fileroot = fileroot
        # self.directory = directory
        self.marshall_signal = marshall_signal #Class
        # self.yolo_name = yolo_name #Class Yolo {Aircraft, Marshaller, Workers}
        self.frame = frame

        #Bounding box
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        # #Videos parameters
        # self.orientation = orientation #{front, back}
        # self.rain = rain
        # self.wind = wind
        # self.snow = snow
        # self.fog = fog
        # self.time = time #{day,night}

        # #Img size
        # self.img_width = img_width
        # self.img_height = img_height        

    # Resize the frame for gesture detection
    def resize_frame(self, frame = 'None'):
        if frame != 'None':
            return self.frame[int(self.y1) : int(self.y2+1), int(self.x1+1) : int(self.x2+1)]
        else:
            # frame = cv2.imread(self.fileroot)
            self.frame = self.frame[int(self.y1) : int(self.y2+1), int(self.x1+1) : int(self.x2+1)]

    # Show info
    def show(self):
        print(f'Name : {self.filename}')
        print(f'Marshall class : {self.marshall_signal}, Yolo class : {self.yolo_name}')
        print(f'Bounding box : [{self.x1}, {self.y1}, {self.x2}, {self.y2}]')
        print(f'Weather : [Snow : {self.snow}, Fog : {self.fog}, Wind : {self.wind}, Rain : {self.rain}]')

    def yolo_normalization(self):
        self.x1 = round(((self.x1 + self.x2) / 2) / self.img_width, 5)
        self.y1 = round(((self.y1 + self.y2) / 2) / self.img_height, 5)
        self.x2 = round(((self.x2 + self.x1) / self.img_width), 5)
        self.y2 = round(((self.y2 + self.y1) / self.img_height), 5)

    def write_txt_label(self, marshall_classification = False, signal_classification = False, path_final = 'None'):
        # Randomly separate the labels train/val
        # Write them on the Yolov7 folders from .XML and .PNG files in dataset
        
        path_to_yolo = 'yolov7/data/'+path_final+'/'

        #Write PNG in Yolo
        shutil.copy('dataset\\videos\\'+self.fileroot[15:-4]+'.jpg', path_to_yolo+self.filename)

        #Normalize the label for Yolo
        self.yolo_normalization()

        with open(path_to_yolo + self.filename[:-4]+'.txt', 'a') as f:

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

    def detect(self, weight = 'yolov7/yolov7.pt', conf = 0.4, image = 'test/image.png', image_cv2 = "None"):
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
class Video():

    def __init__(self, filename = 'None', output_folder = "image_to_detect/", 
                 writing_folder = 'None',
                 clean_txt = False, clean_jpg = False,
                 interface = Interface(),
                 development = False):
        
        self.filename = filename        
        self.output_folder = output_folder + '/' + writing_folder + '/'
        self.writing_folder = writing_folder
        self.bbox_list = pd.DataFrame({'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []})
        
        # Add interface
        self.interface = interface

        # Clean previous files
        if clean_txt :
            self.clean_previous_txt()

        if clean_jpg :
            self.clean_previous_jpg()

        # Store the video to a list of frames
        self.frames = []
        self.frame_count = 0     

        if development :
        
            # Get the video properties
            cap = cv2.VideoCapture(self.filename)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 10 Frames per second
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = 1/fps*10                    

            # Loop through the frames in the video
            while cap.isOpened():
                # Read the current frame
                ret, frame = cap.read()
                time_stamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # If we have reached the end of the video, break the loop
                if not ret:
                    break

                if self.frame_count > 10:
                    break

                if time_stamp > interval * self.frame_count:

                    # Add the current frame to the list of frames
                    filename = os.path.join(self.output_folder, self.filename.split("\\")[-1].split(".")[0] + f"_frame_{self.frame_count}.jpg")
                    cv2.imwrite(filename, frame)
                    self.frames.append(Frame(
                        fileroot=os.path.join(self.output_folder, self.filename.split("\\")[-1].split(".")[0] + f"_frame_{self.frame_count}.jpg"),
                        img_width=width, 
                        img_height=height                   
                    ))

                    # Increment the frame count
                    self.frame_count += 1           

            # Release the video capture object and close all windows
            cap.release()
            cv2.destroyAllWindows()      

    def clean_previous_txt(self):

        folder_path = 'image_to_detect/'
        txt_files = glob.glob(os.path.join(folder_path, '**/*.txt'), recursive=True)

        # iterate over the files and delete them
        for file_path in txt_files:
            os.remove(file_path)

    def clean_previous_jpg(self):

        folder_path = 'image_to_detect/'
        jpg_files = glob.glob(os.path.join(folder_path + self.writing_folder + '/', '*.jpg'), recursive=True)
        
        # iterate over the files and delete them
        for file_path in jpg_files:
            os.remove(file_path)   

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

            self.x1 = 0
            self.x2 = int(self.frames[0].img_width)
            self.y1 = 0
            self.y2 = int(self.frames[0].img_height)

        if not loading_data:

            # Resize the video to this shape and detect the posture
            self.posture()

            # Classify the movement
            self.classify()

            # Kill the thread
            return

    def show(self):
        print(f'Video bbox : (xmin, xmax, ymin, ymax) = ({self.x1}, {self.x2}, {self.y1}, {self.y2})')

    def update(self, frame = None, ite = 0):

        if isinstance(frame, list):
            nb_frames = len(frame)
        else :
            nb_frames = 1

        print(len(self.frames))

        if len(self.frames) >= 14:

            self.frames = self.frames[nb_frames:]
            self.frames.append(Frame(frame = frame, fileroot = frame_name)) 

        else :

            frame_name = f"test_videos/{self.interface.eventID}_{ite}.png"
            # cv2.imwrite(frame_name, frame)
            self.frames.append(Frame(frame = frame, fileroot = frame_name))        

    def classify(self, list_input = None):

        if list_input is not None:
            self.video_angles = list_input

        if len(self.video_angles) == 14:

            try :

                # Predict with the model
                print(np.shape(self.video_angles))
                self.video_angles = np.reshape(self.video_angles, (1, 14, 4))
                output = CLASSIFIER.predict(self.video_angles)
                value = np.argmax(output, axis=1)

                # Generate Kafka exception
                self.interface.json_output(value)

            except Exception as e:

                print(e)


    def posture(self, write_files_norm = False, write_files_angles = False):

        # List of values
        self.video_angles = []
        # self.video_norm = []

        # Output folders
        output_angles = "image_to_detect/angles/"
        output_norm = "image_to_detect/norm/"

        for frame in self.frames:

            frame.x1 = self.x1
            frame.x2 = self.x2
            frame.y1 = self.y1
            frame.y2 = self.y2

            frame.resize_frame()

            # Call Posture function
            pipe = PipeDetection(frame.frame, fileroot = frame.fileroot.split("\\")[-1].split(".")[0])
            self.video_angles.append(pipe.get_angles())
            # self.video_norm.append(pipe.get_landmarks_normalized())

        if write_files_norm:

            # Write TXT norm
            m_norm = np.array(self.video_norm)
            m_norm = m_norm.reshape(-1, m_norm.shape[-1])
            np.savetxt(os.path.join(output_norm, self.filename.split("\\")[-1].split(".")[0] + '.txt'), m_norm, fmt='%.4f') 

        if write_files_angles :

            # Write TXT angles
            m_angles = np.array(self.video_angles)
            m_angles = m_angles.reshape(-1, m_angles.shape[-1])
            np.savetxt(os.path.join(output_angles, self.filename.split("\\")[-1].split(".")[0] + '.txt'), m_angles, fmt='%.4f')

if __name__ == '__main__':

    video = Video()
    print(np.shape([[0,0,0,0] for _ in range(14)]))
    video.classify(list_input=[[0,0,0,0] for _ in range(14)])