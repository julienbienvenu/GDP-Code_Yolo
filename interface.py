import json
import cv2
import random

class Interface:
    # Receive RTSP_URL and output path as parameters
    def __init__(self, rtsp_url, output_path='./video/Videos_of_gestures.avi'):
        self.rtsp_url = rtsp_url
        self.output_path = output_path
        self.eventID = random.randint(100000, 999999)

    # Perform the actual video recording and saving operation
    def video_input(self, rstp_url = ''):

        if rstp_url != '':
            self.rtsp_url = rstp_url

        # Capture video
        cap = cv2.VideoCapture(self.rtsp_url) # Turn on RTSP video streaming
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out = cv2.VideoWriter(self.output_path, fourcc, 24.0, (int(cap.get(3)), int(cap.get(4)))) 
        # Write the processed video stream to a local file. 24 frames per second, get the resolution of the video from the cv2.VideoCapture object cap.


        # Reads video from the video stream frame by frame, writes each frame read to the output file and 
        #   displays the video stream in the window in real time. 
        #   When the user presses the 'q' key, reading and displaying the video stream is stopped.
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow('Video Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def json_output(self, num_class):

        class_names = ['Front' for _ in range(15)]

        json_obj = {
            "EventId": self.eventID,
            "AlarmType": "Conflict",
            "Description": "Conflict is happening",
            "SensorId": "Camera1",
            "Priority": 1,
            "ClassNumber": num_class,
            "ClassName": class_names[num_class - 1]
        }

        return json.dumps(json_obj)
    
if __name__ == '__main__' :

    rtsp_url = "rtsp://username:password@ip_address:port/path" # Camera rstp_url (need to change when we have real camera rstp_url)
    output_path = './video/Videos_of_gestures.avi' # Video stream storage location

    recorder = Interface(rtsp_url, output_path)
    recorder.record_video()
