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

        # REsize image with Yolo output
        # image = self.frame.resize_frame(image)

        # cv2.imshow("Image", image)

        results = pose.process(image)
        cpt = 0
        positions = []

        for landmark in results.pose_landmarks.landmark:
            cpt += 1
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            positions.append([x,y])

            with open('image_to_detect/points.txt', 'a') as file:
                    file.write(f"{cpt}: ({round(x, 5)}, {round(y, 5)})" + '\n')
                    file.close()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def distance_x(self, x):  

        return (x-self.x_ref)/self.x_norm
    
    def distance_y(self, y):
        return (y-self.x_ref)/self.x_norm

    def generate_txt(self, nb_frames = 48):

        path = 'dataset/videos/'
        folders = [f2.path for f1 in os.scandir(path) for f2 in os.scandir(f1.path) if f2.name.endswith(".mp4")]

        # Initialize the Mediapipe drawing objects
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        for h in range(len(folders)) :

            video_path = folders[h]

            video = cv2.VideoCapture(video_path)
            df = np.zeros((33, nb_frames, 2))
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

                # Detect the pose landmarks using Mediapipe
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    results = pose.process(image)

                    # Extract the pose landmarks and store them in the dataframe
                    if results.pose_landmarks is not None:
                        for j, landmark in enumerate(results.pose_landmarks.landmark):
                            df[j][i][0] = landmark.x
                            df[j][i][1] = landmark.y

                    # Draw the pose landmarks on the image
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # We normalize all the values with the following :
                # (24,23) -> x absciss
                # (24,12) -> y absciss

                self.x_norm = abs(df[23][i][0] - df[24][i][0])
                self.y_norm = abs(df[12][i][1] - df[24][i][1])

                self.x_ref = df[24][i][0]
                self.y_ref = df[24][i][1]

                # Normalization
                for j in range(len(df)):
                    df[j][i][0] = round(self.distance_x(df[j][i][0]),3)
                    df[j][i][1] = round(self.distance_y(df[j][i][1]),3)

            # Write the dataframe to the output file
            np.savetxt("PoseEstimation/txt_files/" + txt_file, df.reshape(33, nb_frames*2), fmt='%f')
            print(f'Done {h/len(folders)}: {video_path}')
        