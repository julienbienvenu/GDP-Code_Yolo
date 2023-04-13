# Generate the folders
import glob
import os
import cv2
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter

from matplotlib import pyplot as plt

def test():
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate some example data
    x = np.linspace(-5, 5, 100)  # Generate 100 x values from -5 to 5
    y = np.linspace(-5, 5, 100)  # Generate 100 y values from -5 to 5
    x, y = np.meshgrid(x, y)  # Create a grid of x and y values
    z = np.sin(np.sqrt(x**2 + y**2))  # Calculate z using a wave-like function

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

    # Set labels for the axes
    ax.set_xlabel('Variable 1')
    ax.set_ylabel('Variable 2')
    ax.set_zlabel('Accuracy')

    # Show the plot
    plt.show()



def histo():

    path = "dataset/"
    folders_ytb = glob.glob(os.path.join(path, 'videos/**/*.mp4'), recursive=True)
    folders_shot = glob.glob(os.path.join(path, 'shot_videos/**/*.mp4'), recursive=True)
    class_indices_ytb = []
    class_indices_shot = []

    for file in folders_ytb:
        class_indices_ytb.append(int(file.split('\\')[-1].split('.')[0].split('_')[-2]))

    for file in folders_shot:
        class_indices_shot.append(int(file.split('\\')[-1].split('.')[0].split('_')[-2]))

    class_counts_ytb = {}
    for label in class_indices_ytb:
        if label not in class_counts_ytb:
            class_counts_ytb[label] = 0
        class_counts_ytb[label] += 1

    class_counts_shot = {}
    for label in class_indices_shot:
        if label not in class_counts_shot:
            class_counts_shot[label] = 0
        class_counts_shot[label] += 1
        
    fig, ax = plt.subplots()

    hist_ytb, bins_ytb, patches_ytb = plt.hist([class_indices_ytb, class_indices_shot], bins = 15, label = ['YTB', 'SHOT', 'AUG'], stacked=True)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Histogram of Classes in Dataset')
    plt.xticks([(i+1) for i in range(15)])
    plt.legend()
    plt.savefig('output_graph/dataset.png')

    # Set the x-axis labels under the middle of the bins with a shift of 0.5 units
    bin_centers_ytb = 0.5 * np.diff(bins_ytb) + bins_ytb[:-1]
    ax.set_xticks(bin_centers_ytb)
    ax.set_xticklabels([(i+1) for i in range(15)], ha='center')
    ax.xaxis.set_tick_params(pad=15) 
    plt.show()

def pipe_0_ratio():

    folder = glob.glob(os.path.join('image_to_detect/angles/*.txt'), recursive=True)
    zero_ratio = []

    for file in folder :
        # Read the text file into a pandas DataFrame
        df = pd.read_csv(file, header=None, sep = ' ')

        # Count the ratio of 0s in the DataFrame
        num_zeros = (df == 0).any(axis=1).sum()
        if num_zeros != 0:
            zero_ratio.append(num_zeros / len(df))    

    
    # Create a histogram
    plt.hist([zero_ratio], bins=20)
    plt.xlabel('Zero Ratio')
    plt.ylabel('Frequency')
    plt.title('Line with Zero Ratio in Matrix')
    plt.show()

def test_video():
    # Get the video properties
    cap = cv2.VideoCapture('test_videos/test.mp4')

    # 10 Frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1/fps*10

    # Load model
    frame_count = 0

    frames = []

    # Loop through the frames in the video
    while cap.isOpened():

        if frame_count > 55:
            break
        
        # Read the current frame
        ret, frame = cap.read()
        time_stamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if time_stamp > interval * frame_count:

            # Add frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # Draw the bounding box on the image
            x1, x2, y1, y2 = 200, 400, 50, 400
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the image with the bounding box
            cv2.imwrite('photo.png', frame_rgb)

            frames.append(frame_rgb)
            
            frame_count += 1    
            print(frame_count)

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

    return frames

if __name__ == '__main__':
    test_video()
    # test()