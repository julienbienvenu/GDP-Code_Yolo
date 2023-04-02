import glob
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import time

class OpticalFlow():

    def __init__(self):

        pass

    def generate(self):

        # Define resize parameters
        width = 640
        height = 360

        path = "dataset/"
        folders = glob.glob(os.path.join(path, '**/*.mp4'), recursive=True)

        for file in folders :
            
            # Load video and extract first frame
            cap = cv2.VideoCapture(file)
            ret, frame1 = cap.read()
            frame1 = cv2.resize(frame1, (width, height))

            # Set the frame rate to 30 frames per second
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            duration = 1  # Duration of the optical flow in seconds
            num_frames = int(frame_rate * duration)

            # Convert first frame to grayscale
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            # Set accumulator for optical flow
            acc_flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
            start = time.time()

            for i in range(num_frames):
                # Extract next frame
                ret, frame2 = cap.read()
                if not ret:
                    break
                frame2 = cv2.resize(frame2, (width, height))

                # Convert current frame to grayscale
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                # Compute optical flow using Farneback method
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
                                                    cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Accumulate optical flow
                acc_flow += flow

                # Update previous frame
                prvs = next


            # Compute magnitude and direction of accumulated flow
            mag, ang = cv2.cartToPolar(acc_flow[..., 0], acc_flow[..., 1])

            # Save magnitude and direction as text files
            name = file.split('\\')[-1].split('.')[0]
            np.savetxt('image_to_detect/opt_flow/magnitude/' + name + '.txt', mag.flatten())
            np.savetxt('image_to_detect/opt_flow/direction/' + name + '.txt', ang.flatten())

            cap.release()

            elapsed_time = time.time() - start
            print(f'{file} : {elapsed_time:.2f} s.')

    def train(self):

        data_dir = 'image_to_detect/opt_flow/'
        
        # Loop over each video and extract features and labels
        X_list = []
        y_list = []
        for i, mag_file in enumerate(os.listdir(os.path.join(data_dir, 'magnitude'))):
            if not mag_file.endswith('.txt'):
                continue

            filename = mag_file.split('\\')[-1].split('.')[0]
            mag = np.loadtxt(os.path.join(data_dir, 'magnitude', mag_file))
            dir_file = os.path.join(data_dir, 'direction', filename)
            y = int(filename.split('_')[-2])
            ang = np.loadtxt(dir_file + '.txt')
            X = np.column_stack((mag, ang))
            X_list.append(X)
            y_list.append(y * np.ones(X.shape[0], dtype=int))

        # Concatenate features and labels from all videos into a single array
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train SVM classifier using grid search with cross-validation
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }

        print('Run SVC')
        clf = SVC()
        grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='accuracy', verbose = 2)
        grid_search.fit(X_train, y_train)

        # Plot accuracy over time
        plt.plot(grid_search.cv_results_['mean_train_score'], label='Train')
        plt.plot(grid_search.cv_results_['mean_test_score'], label='Validation')
        plt.legend()
        plt.xlabel('Grid search fold')
        plt.ylabel('Accuracy')
        plt.savefig('output_graph/Optical_Flow/svc_train.png')
        plt.show()

        # Evaluate classifier on testing data
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy: {:.2f}%'.format(accuracy * 100))

if __name__ == '__main__':

    flow = OpticalFlow()
    # flow.generate()
    flow.train()