import cv2
import mediapipe as mp

from frame import Frame

class PostDetection():

    def __init__(self, frame = Frame()):
        self.frame = frame

    def detection(self):

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        image = cv2.imread(self.frame.fileroot)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # REsize image with Yolo output
        # image = self.frame.resize_frame(image)

        # cv2.imshow("Image", image)

        results = pose.process(image)

        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            print(f"Landmark {landmark}: ({x}, {y})")

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        