import glob
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time
import os

from sklearn.tree import DecisionTreeClassifier

class OpticalFlow():

    def __init__(self):

        pass

    def data_augmentation(self):

        pass

        print(f'Data augmentation : start / X : {self.X_train.shape}, y : {self.y_train.shape}')

        max_percent_change = 0.4
        max_size_change = 0.8
        new_x_list = []
        new_y_list = []

        # Convert one-hot encoded labels to class indices
        class_indices = np.argmax(self.y_train, axis=1)

        # Count the number of instances for each class in the original dataset
        class_counts = {}
        for label in class_indices:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Determine the maximum number of instances for each class in the augmented dataset
        # We multiply the dataset by 3
        max_class_count = max(class_counts.values()) * 50

        for label, count in class_counts.items():
            # Add the original matrices to the new list
            label_indices = np.where(class_indices == label)[0]
            arrs = self.X_train[label_indices]
            new_x_list.extend(arrs)
            new_y_list.extend([self.y_train[label_indices[0]]] * count)
            # Determine how many more instances to add for this class
            num_to_add = max_class_count - count

            # Create new matrices with 10% maximum variation
            for _ in range(num_to_add):
                # Randomly select an original matrix to modify
                arr = arrs[np.random.randint(len(arrs))]

                # Copy the original matrix
                new_arr = np.copy(arr)

                # Determine the number of elements to change
                num_elements = int(np.round(arr.size * max_size_change))

                # Create a mask to select the elements to change
                mask = np.zeros_like(arr, dtype=bool)
                mask[np.random.choice(arr.shape[0], size=num_elements, replace=True),
                    np.random.choice(arr.shape[1], size=num_elements, replace=True)] = True

                # Modify the selected elements
                max_change =  np.abs(arr[mask]) * max_percent_change
                new_arr[mask] = np.random.uniform(low=arr[mask] - max_change, high=arr[mask] + max_change)

                # Add the new matrix to the list
                new_x_list.append(new_arr)
                new_y_list.append(self.y_train[label_indices[0]])

        self.X_train = np.array(new_x_list)
        self.y_train = np.array(new_y_list)

        print(f'Data augmentation : end / X : {self.X_train.shape}, y : {self.y_train.shape}')

        # Plot a histogram to verify the number of instances for each class in the augmented dataset
        print(class_indices)
        plt.hist(class_indices, bins=len(class_counts))
        plt.xticks(list(class_counts.keys()))
        plt.xlabel('Class')
        plt.ylabel('Number of Instances')
        plt.title('Histogram of Classes in Dataset')
        plt.savefig('output_graph/dataset_before_augmentation.png')
        plt.show()

        # Convert one-hot encoded labels to class indices
        class_indices = np.argmax(self.y_train, axis=1)

        # Count the number of instances for each class in the augmented dataset
        class_counts = {}
        for label in class_indices:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Update the histogram with the new counts
        plt.clf()
        plt.hist(class_indices, bins=len(class_counts))
        plt.xticks(list(class_counts.keys()))
        plt.xlabel('Class')
        plt.ylabel('Number of Instances')
        plt.title('Histogram of Classes in Augmented Dataset')
        plt.savefig('output_graph/dataset_after_augmentation.png')
        plt.show()

    def generate(self, directory = 'opt_flow', duration = 1, nb_frames = 10, folders = []):

        # Define resize parameters
        width = 640
        height = 360
        time_avg = 0

        if len(folders) == 0:
            path = "dataset/"
            folders = glob.glob(os.path.join(path, '**/*.mp4'), recursive=True)

            path_output = "image_to_detect/"
            folders_output = glob.glob(os.path.join(path_output, f'{directory}_{duration}_{nb_frames}/direction/*.txt'), recursive=True)
            folders_output = [file.split('\\')[-1].split('.')[0] for file in folders_output]

            dataset_size = len(folders_output)

            for file in folders[:]:
                if file.split('\\')[-1].split('.')[0] in folders_output:
                    folders.remove(file)

            print(f'Files to process : {len(folders)} ({round((len(folders)/dataset_size), 2)} %)')

        for file in folders :
            
            # Load video and extract first frame
            cap = cv2.VideoCapture(file)
            ret, frame1 = cap.read()
            frame1 = cv2.resize(frame1, (width, height))

            # Set the frame rate to 30 frames per second
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            duration = duration  # Duration of the optical flow in seconds
            num_frames = int(frame_rate * duration)
            interval = int(frame_rate / nb_frames)  # Number of frames to skip between each optical flow calculation

            # Convert first frame to grayscale
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            # Set accumulator for optical flow
            acc_flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
            start = time.time()

            for i in range(num_frames):
                # Extract next frame
                for j in range(interval - 1):
                    ret, _ = cap.read()
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
            np.savetxt(f'image_to_detect/{directory}_{duration}_{nb_frames}/magnitude/' + name + '.txt', mag.flatten())
            np.savetxt(f'image_to_detect/{directory}_{duration}_{nb_frames}/direction/' + name + '.txt', ang.flatten())

            cap.release()

            elapsed_time = time.time() - start
            print(f'{file} : {elapsed_time:.2f} s.')

        self.time = time_avg/len(folders)

    def load_data(self, directory = 'opt_flow'):

        data_dir = 'image_to_detect/' + directory + '/'
        
        # Loop over each video and extract features and labels
        X_list = []
        y_list = []

        print('Generating list')

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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Add data augmentation
        # self.data_augmentation()

    def train(self, directory = 'opt_flow'):

        self.load_data(directory=directory)

        epochs = 20  

        print('Running models ...')
        # Define the preweighted models
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=epochs),
            # 'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=epochs),
            # 'SVM': SVC(class_weight='balanced', max_iter=epochs),
            # 'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=epochs),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=3),
            # 'KNN': KNeighborsClassifier(weights='distance', n_neighbors=10),
            # 'Multi-layer Perceptron': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=epochs),
            # 'AdaBoost': AdaBoostClassifier(n_estimators=epochs)
        }

        # Train and evaluate each model for X epochs
        scores = []
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            score = accuracy_score(self.y_test, y_pred)
            scores.append(score)
            print(f"{name}: {score}")

        # Plot the results in a histogram
        fig, ax = plt.subplots()
        ax.bar(models.keys(), scores)
        ax.set_ylabel('Accuracy')
        ax.set_xticklabels(models.keys(), rotation=45, ha='right')
        plt.title(f"Accuracy Scores for {epochs} epochs")
        plt.tight_layout()
        plt.savefig(f'output_graph/Optical_Flow/{directory}_t_{self.time}.png')
        plt.clf()


def flow_and_generate_period():

    # We create a repository with 10 flows per class for different time period 
    # save the average computation time

    flow = OpticalFlow()
    
    # Parameters
    timestamps = [1, 2, 3]
    precisions = [3, 6, 9, 12]

    # Generate the folders
    path = "dataset/"
    folders = glob.glob(os.path.join(path, '**/*.mp4'), recursive=True)

    iterations = np.zeros(15)

    for file in folders[:]:
        if iterations[int(file.split('\\')[-1].split('.')[0].split('_')[-2]) - 1] >= 5:
            folders.remove(file)
        else :
            iterations[int(file.split('\\')[-1].split('.')[0].split('_')[-2]) - 1] += 1

    # Loop
    for time in timestamps:
        for precision in precisions:

            try :
                folder_name = f"image_to_detect/opt_flow_{time}_{precision}"
                os.makedirs(folder_name)
                os.makedirs(os.path.join(folder_name, "direction"))
                os.makedirs(os.path.join(folder_name, "magnitude"))            

                flow.generate(duration = time, nb_frames = precision, folders = folders)
                flow.train(directory = f'opt_flow_{time}_{precision}')
            except :
                flow.train(directory = f'opt_flow_{time}_{precision}')

def replace_nan_inf(x):
    inf_indices = np.isinf(x)
    x[inf_indices] = 10
    x = np.nan_to_num(x)
    return x

if __name__ == '__main__':

    flow_and_generate_period()
    # flow.generate()
    # flow.train()