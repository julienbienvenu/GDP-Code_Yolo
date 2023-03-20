import cv2
import mediapipe as mp
import os
import numpy as np

from frame import Frame

class PoseDetection():

    def __init__(self, frame = Frame()):
        self.frame = frame

        self.x_norm = 0
        self.y_norm = 0

        self.x_ref = 0
        self.y_ref = 0

    def detection(self):

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        image = cv2.imread(self.frame.fileroot)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image with Yolo output
        image = self.frame.resize_frame(image)

        # cv2.imshow("Image", image)

        results = pose.process(image)
        cpt = 0
        points_list = [11,12,13,14,15,16,23,24]
        
        df = np.zeros((8, 1, 2))

        if results.pose_landmarks is not None:

            # Detect the pose landmarks using Mediapipe
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                results = pose.process(image)

                # Extract the pose landmarks and store them in the dataframe
                if results.pose_landmarks is not None:
                    cpt = 0
                    i=0
                    for j, landmark in enumerate(results.pose_landmarks.landmark):

                        if j in points_list : 

                            df[cpt][i][0] = landmark.x
                            df[cpt][i][1] = landmark.y
                            cpt += 1
                            cv2.circle(image, (int(landmark.x), int(landmark.y)), 15, (0, 255, 0), -1)

                    self.x_norm = abs(df[6][i][0] - df[7][i][0])
                    self.y_norm = abs(df[1][i][1] - df[7][i][1])

                    self.x_ref = df[7][i][0]
                    self.y_ref = df[7][i][1]

                    # Normalization
                    for j in range(len(df)):
                        df[j][i][0] = round(self.distance_x(df[j][i][0]),3)
                        df[j][i][1] = round(self.distance_y(df[j][i][1]),3)

                    # Write the dataframe to the output file
                    np.savetxt('image_to_detect/points.txt', df.reshape(8, 2), fmt='%f')                

        else : 

            print(f'No detection : {self.frame.fileroot}')

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image with MediaPipe points", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def distance_x(self, x):  

        return (x-self.x_ref)/self.x_norm
    
    def distance_y(self, y):
        return (y-self.x_ref)/self.x_norm

    def generate_txt(self, nb_frames = 48):

        path = 'dataset/videos/'
        folders = [f2.path for f1 in os.scandir(path) for f2 in os.scandir(f1.path) if f2.name.endswith(".mp4")]
        
        points_list = [11,12,13,14,15,16,23,24]

        # Initialize the Mediapipe drawing objects
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        for h in range(len(folders)) :

            video_path = folders[h]           

            video = cv2.VideoCapture(video_path)
            df = np.zeros((8, nb_frames, 2))
            txt_file = video_path.split('\\')[-1][:-4] + '.txt'

            for i in range(nb_frames):

                # Read the frame
                success, image = video.read()

                # If there are no more frames, fill the remaining frames with zeros and break out of the loop
                if not success:
                    df[:, :, i:] = 0
                    break

                # Flip the image horizontally for correct display
                image = cv2.flip(image, 1)

                # Convert the image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Load frame and resize image 
                self.frame = Frame()
                self.frame.detect(image_cv2 = image)
                image = self.frame.resize_frame(image)

                # Detect the pose landmarks using Mediapipe
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    results = pose.process(image)

                    # Extract the pose landmarks and store them in the dataframe
                    if results.pose_landmarks is not None:
                        cpt = 0
                        for j, landmark in enumerate(results.pose_landmarks.landmark):

                            if j in points_list : 
                                
                                df[cpt][i][0] = landmark.x
                                df[cpt][i][1] = landmark.y
                                cpt += 1

                # We normalize all the values with the following :
                # (24,23) -> x absciss
                # (24,12) -> y absciss

                self.x_norm = abs(df[6][i][0] - df[7][i][0])
                self.y_norm = abs(df[1][i][1] - df[7][i][1])

                self.x_ref = df[7][i][0]
                self.y_ref = df[7][i][1]

                # Normalization
                for j in range(len(df)):
                    df[j][i][0] = round(self.distance_x(df[j][i][0]),3)
                    df[j][i][1] = round(self.distance_y(df[j][i][1]),3)

            # Write the dataframe to the output file
            np.savetxt("PoseEstimation/txt_files/" + txt_file, df.reshape(8, nb_frames*2), fmt='%f')
            print(f'Done {h/len(folders)}: {video_path}')
        