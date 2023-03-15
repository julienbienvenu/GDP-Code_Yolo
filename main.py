# Main file
import cv2
import os
import numpy as np
import shutil
from PIL import Image
import random
from PoseEstimation.pose_pipe import PoseDetection
from frame import Frame

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

#Main function
def main():    
    
    # Get path to .avi
    path_to_avi = get_avi()

    #We create a program to only get 15% of the images
    for path in path_to_avi:

        if 1==1:
            # Load the video
            cap = cv2.VideoCapture(path)

            # Get the video frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Set the start and end frame
            start_frame = 0
            end_frame = total_frames

            print(f"Total frames : {end_frame}")

            # Loop through the frames
            for i in range(start_frame, end_frame):
                # Read a frame
                ret, frame = cap.read()

                # Break the loop if the frame is not retrieved
                if not ret:
                    break

                # Mirror the frame along the x-axis
                frame = cv2.resize(cv2.flip(frame, 1), (80,60), interpolation=cv2.INTER_LINEAR)

                # Save the mirrored frame as an image
                cv2.imwrite("data_img/"+path[-11:-4]+"_{}.jpg".format(i), frame)

        # Release the video capture
        cap.release()


if __name__ == "__main__":
    #Update : 12/02/2023 18:20
    # main()
    # create_txt(path_img)
    # send_to_yolo()
    frame = Frame(0,0,0,0,fileroot = 'image_to_detect/image.jpg', img_height=720, img_width=1280)
    frame.detect()
    frame.show()

    pose = PoseDetection(frame = frame)
    pose.detection()
