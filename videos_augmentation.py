import glob
import cv2
import numpy as np
import os

def video_augmentation():  

    # Set parameters for augmentation
    rotation_angles = [60]
    brightness_reductions = [505]
    rain_strengths = [0.50]

    # Set the path to the folder containing the videos
    folders = glob.glob(os.path.join("dataset/", '**/*.mp4'), recursive=True)

    # Loop through all the video files in the folder
    for file_name in folders:

        print(file_name)
        cpt = 0

        for rotation_angle in rotation_angles:
            for brightness_reduction in brightness_reductions :
                for rain_strength in rain_strengths :

                    cpt += 1

                    # Read the video file                    
                    video = cv2.VideoCapture(file_name)
                    
                    # Get the video properties
                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(video.get(cv2.CAP_PROP_FPS))
                    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Define the output video writer
                    output_file = os.path.join('dataset/da_videos', f'da_{cpt}_' + file_name.split('\\')[-1])
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

                    # Rotate the frame by a random angle
                    angle = np.random.randint(-rotation_angle, rotation_angle)
                    
                    # Loop through all the frames in the video
                    for i in range(num_frames):
                        # Read the frame
                        ret, frame = video.read()
                        
                        # Rotate the frame by a random angle
                        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                        frame = cv2.warpAffine(frame, M, (width, height))
                        
                        # Add rain to the frame
                        noise = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.randn(noise, 0, rain_strength * 255)
                        frame = cv2.add(frame, noise)
                        
                        # Reduce the brightness of the frame
                        frame = cv2.subtract(frame, brightness_reduction)
                        
                        # Write the frame to the output video
                        output_video.write(frame)
                    
                    # Release the video objects and close the output writer
                    video.release()
                    output_video.release()

    return 0

if __name__ == '__main__':
    video_augmentation()