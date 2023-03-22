import cv2
import mediapipe as mp
import numpy as np
import os

class PipeDetection():

    def __init__(self, frame, fileroot):
        self.frame = frame 
        self.fileroot = fileroot 

        self.set_matrix_landmarks()     

    def set_matrix_landmarks(self):
        ### INPUT: an object from the class Frame
        ### OUTPUT: initialize the value of the attribute landmarks with a 8 * 3 matrix with relative coordinates for each relevant landmark in the order: 
        ###         R wrist, R elbow, R shoulder, L shoulder, L elbow, L wrist, R hip, L hip
        # 
        # Perform pose detection after converting the image into RGB format.
        # Initializing mediapipe pose class.
        mp_pose = mp.solutions.pose

        # Setting up the Pose function.
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
        results = pose.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        mat = []
        # Check if any landmarks are found.
        if results.pose_landmarks:
            
            # Iterate to keep only the relevant landmarks
            for i in [16, 14, 12, 11, 13, 15, 24, 23]:
                mat.append([results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z])

        self.landmarks = mat.copy()

    def get_landmarks_normalized(self):
        ### INPUT: an object from the class Frame
        ### OUTPUT: a 8 * 3 matrix with normalized coordinates for each relevant landmark in the order: 
        ###         R wrist, R elbow, R shoulder, L shoulder, L elbow, L wrist, R hip, L hip

        mat = self.landmarks
        matnormalized = []

        for i in range(len(mat)):
            normalized = mat[i].copy()
            normalized[0] = (normalized[0] - mat[6][0]) / (mat[7][0] - mat[6][0])
            normalized[1] = (normalized[1] - mat[6][1]) / (mat[2][1] - mat[6][1])
            normalized[2] = normalized[2] / mat[6][2]
            matnormalized.append(normalized)

        self.matnormalized = matnormalized

        return matnormalized
    
    def get_angles(self):

        ### INPUT: obkect in the class Frame
        ### OUTPUT: a list with four angles: the right wrist, elbow, shoulder angle 
        ###                                  the right elbow, shoulder, hip angle
        ###                                  the left elbow, shoulder, hip angle
        ###                                  the left wrist, elbow, shoulder angle

        image_height, image_width, _ = self.frame.shape
        matpixels = []

        for i in range(len(self.landmarks)):
            pixels = self.landmarks[i].copy()
            pixels[0] = pixels[0]*image_width
            pixels[1] = pixels[1]*image_height
            pixels[2] = pixels[2]*image_width
            matpixels.append(pixels)

        ax = matpixels[0][0] - matpixels[1][0]
        ay = matpixels[0][1] - matpixels[1][1]
        az = matpixels[0][2] - matpixels[1][2]

        bx = matpixels[2][0] - matpixels[1][0]
        by = matpixels[2][1] - matpixels[1][1]
        bz = matpixels[2][2] - matpixels[1][2]

        angle_R_WES = ([ax, ay, az], [bx, by, bz])

        ax = matpixels[1][0] - matpixels[2][0]
        ay = matpixels[1][1] - matpixels[2][1]
        az = matpixels[1][2] - matpixels[2][2]

        bx = matpixels[6][0] - matpixels[2][0]
        by = matpixels[6][1] - matpixels[2][1]
        bz = matpixels[6][2] - matpixels[2][2]

        angle_R_ESH = ([ax, ay, az], [bx, by, bz])

        ax = matpixels[4][0] - matpixels[3][0]
        ay = matpixels[4][1] - matpixels[3][1]
        az = matpixels[4][2] - matpixels[3][2]

        bx = matpixels[7][0] - matpixels[3][0]
        by = matpixels[7][1] - matpixels[3][1]
        bz = matpixels[7][2] - matpixels[3][2]

        angle_L_ESH = ([ax, ay, az], [bx, by, bz])

        ax = matpixels[5][0] - matpixels[4][0]
        ay = matpixels[5][1] - matpixels[4][1]
        az = matpixels[5][2] - matpixels[4][2]

        bx = matpixels[3][0] - matpixels[4][0]
        by = matpixels[3][1] - matpixels[4][1]
        bz = matpixels[3][2] - matpixels[4][2]

        angle_L_WES = ([ax, ay, az], [bx, by, bz])

        angles = []

        for angle_vec in [angle_R_WES, angle_R_ESH, angle_L_ESH, angle_L_WES]:
            # define two vectors
            a = np.array(angle_vec[0])
            b = np.array(angle_vec[1])

            cp = np.cross(a, b)

            # calculate the angle between the two vectors in radians
            angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

            if np.sign(cp[2]) < 0:
                angle = -angle

            # convert the angle to degrees
            angle_degrees = np.degrees(angle)

            angles.append(angle_degrees)

        self.angles = angles

        return angles
    
    def write_txt(self):

        m = self.get_landmarks_normalized()
        a = self.get_angles()

        # Output folders
        output_angles = "image_to_detect/angles/"
        output_norm = "image_to_detect/norm/"

        # Write TXT norm
        m_norm = np.array(self.matnormalized)
        np.savetxt(os.path.join(output_norm, self.fileroot + '.txt'), m_norm, fmt='%.4f') 

        # Write TXT angles
        m_angles = np.array(self.angles)
        np.savetxt(os.path.join(output_angles, self.fileroot + '.txt'), m_angles, fmt='%.4f') 
        