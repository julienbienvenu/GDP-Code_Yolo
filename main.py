# Main file
import cv2
import os
import numpy as np
import shutil
from PIL import Image
import random
from PoseEstimation.pose_pipe import PoseDetection
from frame import Frame, Video

from labels import convert_xml_to_txt

#Different path
path_video = "data_video_tr"
path_img = "data_img"

#Retrive .avi
def get_avi():

    path = path_video    
    subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]
    files = []

    for sub_path in subdirectories:
        files = files + [sub_path+"/"+f for f in os.listdir(sub_path) if f.endswith(".avi")]

    return files

#Create the .txt files for Yolov7
def create_txt(path):

    paths = [path+"/"+f for f in os.listdir(path) if f.endswith(".jpg")]
    
    for i in range(len(paths)):

        if paths[i][11] != "0":
            np.savetxt("data_labels/" + paths[i][9:-4]+'.txt', np.column_stack(np.array([int(paths[i][11]), 0.5, 0.5, 1.0, 1.0])), newline = " ", fmt='%i %f %f %f %f')
        else :
            np.savetxt("data_labels/" + paths[i][9:-4]+'.txt', np.column_stack(np.array([10, 0.5, 0.5, 1.0, 1.0])), newline = " ", fmt='%i %f %f %f %f')

        print(f"Done for {i}/{len(paths)}")

#Generate proper datasets for Yolov7
def send_to_yolo():
    
    dst_train = "yolov7/data/train/"
    dst_test = "yolov7/data/val/"

    image_files = ["data_img/"+f for f in os.listdir("data_img") if f.endswith(".jpg")]
    random.shuffle(image_files)

    train_list = image_files[:int(0.75*len(image_files))]
    test_list = image_files[int(0.75*len(image_files)):]

    for img in test_list:
        shutil.move(img, dst_test + img[9:])
        shutil.move("data_labels/" + img[9:-4]+".txt", dst_test + img[9:-4]+".txt")

    print("0.25 achieve")

    for img in train_list:
        shutil.move(img, dst_train + img[9:])
        shutil.move("data_labels/" + img[9:-4]+".txt", dst_train + img[9:-4]+".txt")    

    print('Done')

def main():
    # This function take a video as input and perform the detection on all its frames
    # No update for the moment

    input_video = Video(filename="image_to_detect/test.mp4")
    input_video.detection()
    input_video.show()
    input_video.posture()

if __name__ == "__main__":
    
    main() 

    # pose = PoseDetection()
    # pose.generate_txt()
