# Main file
import time
import cv2
import os
import numpy as np
import shutil
from PIL import Image
import random
from PoseEstimation.pose_pipe import PoseDetection
from frame import Frame, Video
import glob
from multiprocessing import Pool
from tqdm import tqdm
import copy
import threading
from frame_light import Video_light
from interface import Interface

from labels import convert_xml_to_txt
from test import test_video

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



def main_generate_txt_posture():
    
    print('Hello')

def main_train():
    # This function take a video as input and perform the detection on all its frames
    # No update for the moment

    path = "dataset/"
    folders = glob.glob(os.path.join(path, '**/*.mp4'), recursive=True)

    path_output = "image_to_detect/"
    folders_output = glob.glob(os.path.join(path_output, 'angles/*.txt'), recursive=True)
    folders_output = [file.split('\\')[-1].split('.')[0] for file in folders_output]

    for file in folders[:]:
        if file.split('\\')[-1].split('.')[0] in folders_output:
            folders.remove(file)

    print(len(folders))

    run_detection_list(folders)

    # # We use 4 threads
    # detection_folders = [f'detection_{i+1}' for i in range(8)]

    # sublist_size = (len(folders) + 7) // 8
    # result_folders = [(folders[i*sublist_size:(i+1)*sublist_size], i) for i in range(8)]

    # with Pool(processes = 4) as pool:
    #     results = list(tqdm(pool.imap(run_detection_list, result_folders), total=len(result_folders)))


def run_detection_list(folders):

    folder_list, ite = folders, 1
    l = len(folder_list)
    start = time.time()

    print(l)

    for i in range(25):

        folder = folder_list[i]
        r = round((i+1)/l * 100, 4)
        print(f"{r} %")

        input_video = Video(filename = folder, clean_jpg = True, writing_folder = f'detection_{ite+1}')
        input_video.detection(yolo = False, loading_data=True)
        input_video.posture(write_files_angles=True)

        # del input_video


def main_test():

    # Create Interface instance
    interface = Interface()

    # Create Video instance
    video = Video_light(interface = interface)

    # Start the process
    print('Start the process')

    # Iteration initialisation
    ite = 0
    frames = test_video()

    for frame in frames:

        # Simulating frame input, replace this with your actual frame input logic
        # frame = interface.video_input()

        if frame is not None:

            # Ite increasing
            ite = (ite + 1) % 999

            # Run Video.update(frame) on the main thread
            video.update(frame = frame, ite = ite)

            # Copy the Video object and update the eventID interface
            # video_copy = copy.deepcopy(video)
            # video_copy.interface.eventID = video_copy.interface.eventID * 1000 + ite

            video.detection()

            # Create a new thread to run Video.detect()
            # detect_thread = threading.Thread(target = video_copy.detection())
            # detect_thread.start()

def main():

    # Create Interface instance
    interface = Interface()

    # Create Video instance
    video = Video(interface = interface)

    # Iteration initialisation
    ite = 0

    while True:

        # Simulating frame input, replace this with your actual frame input logic
        frame = interface.video_input()

        if frame is not None:

            # Ite increasing
            ite = (ite + 1) % 999

            # Run Video.update(frame) on the main thread
            video.update(frame)

            # Copy the Video object and update the eventID interface
            video_copy = copy.deepcopy(video)
            video_copy.interface.eventID = video_copy.interface.eventID * 1000 + ite

            # Create a new thread to run Video.detect()
            detect_thread = threading.Thread(target = video_copy.detection())
            detect_thread.start()

if __name__ == "__main__" :
    
    main_test()

    # pose = PoseDetection()
    # pose.generate_txt()