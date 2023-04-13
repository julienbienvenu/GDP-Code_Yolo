import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
import os
import pandas as pd
from keras.models import save_model, load_model
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from keras.regularizers import L2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import TimeseriesGenerator
from numpy.random import randint
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, models
import numpy as np

ADD_RATIO = 3

class FCNN_Model():

    def __init__(self, X = [], y = [], load_model = False, max_percent_change = 0.6, max_size_change = 0.6, plot = True):

        self.plot_bool = plot

        if load_model == False :

            # model = Sequential()
            # model.add(Flatten(input_shape=(14, 4)))
            # model.add(Dense(16, activation='relu'))
            # model.add(BatchNormalization())
            # model.add(Dropout(0.5))
            # model.add(Dense(8, activation='relu'))
            # model.add(BatchNormalization())
            # model.add(Dropout(0.5))
            # model.add(Dense(8, activation='relu'))
            # model.add(BatchNormalization())
            # model.add(Dense(15, activation='softmax'))

            model = Sequential()
            model.add(LSTM(14, input_shape=(14, 4), return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.8))
            model.add(LSTM(16, return_sequences=True, kernel_regularizer=L2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.8))
            model.add(LSTM(16, return_sequences=False))
            model.add(Dense(15, activation='softmax'))
            print(model.summary())

            # Compile the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model = model
            

        else :

            self.load_model()        

        if len(X) != 0:
            # Dataset
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

        else :
            # Load Data
            self.load_training_data()
            
            # Dataset
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.15, random_state = 42)

            # We perform some data augmention
            self.data_augmentation(max_percent_change, max_size_change)

    def train(self, epochs=200, batch_size=32):

        # Train the model on your data
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=epochs, batch_size=batch_size)
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'output_graph/accuracy_batch_{batch_size}.png')

        save_model(self.model, f'PoseEstimation/fcnn_batch{batch_size}.h5')

        return history.history['accuracy']

    def train_cross_validation(self):
        # define the number of folds and the batch size
        k = 5
        batch_size = 64

        # create the cross-validation folds using KFold
        kf = KFold(n_splits=k, shuffle=True)

        # loop over the folds and train/evaluate the model
        for train_index, test_index in kf.split(self.X):
            # get the train and test data for this fold
            x_train, y_train = self.X[train_index], self.y[train_index]
            x_test, y_test = self.X[test_index], self.y[test_index]

            # train the model for this fold
            self.model.fit(x_train, y_train, epochs=4, batch_size=batch_size, verbose=0)

            # evaluate the model on the test data for this fold
            score = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
            print('Fold accuracy:', score[1])

    def data_augmentation(self, max_percent_change = 0.6, max_size_change = 0.6):

        print(f'Data augmentation : start / X : {self.X_train.shape}, y : {self.y_train.shape}')

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
        max_class_count = max(class_counts.values()) * ADD_RATIO

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
        plt.hist(class_indices, bins=15)
        plt.xticks([i+1 for i in range(15)])
        plt.xlabel('Class')
        plt.ylabel('Number of Instances')
        plt.title('Histogram of Classes in Dataset')
        plt.savefig('output_graph/dataset_before_augmentation.png')

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
        plt.hist(class_indices, bins=15)
        plt.xticks([i+1 for i in range(15)])
        plt.xlabel('Class')
        plt.ylabel('Number of Instances')
        plt.title('Histogram of Classes in Augmented Dataset')
        plt.savefig('output_graph/dataset_after_augmentation.png')
        plt.clf()

    def load_model(self):

        self.model = load_model('PoseEstimation/fcnn.h5')
        print('Model is successfully loaded')

    def load_training_data(self):

        print(f'Collecting data')

        X = []
        y = []
        
        path = "image_to_detect/angles/"
        files = [path+f for f in os.listdir(path) if f.endswith(".txt")]

        for file in files:
            y_p = [0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            df = pd.read_csv(file, delim_whitespace=True, header=None)

            # Modify NaN values
            df = df.fillna(0)

            # Fill lines
            num_rows_to_add = 14 - df.shape[0]

            # rows_to_concat = []
            # for i in range(num_rows_to_add):
            #     row = pd.Series([i, i+1, i+2, i+3], index=df.columns)
            #     rows_to_concat.append(row)

            # df = pd.concat([df] + rows_to_concat, ignore_index=True)

            for i in range(num_rows_to_add):
                row = [i, i+1, i+2, i+3]  # example data
                df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

            X.append(df.values)
            y_p[int(file.split('_')[-2]) - 1] = 1
            y.append(y_p)

        self.X = np.array(X)
        self.y = np.array(y)

        # Data Augmentation
        # self.X, self.y = data_augmentation(self.x, self.y)

        print(self.X.shape, self.y.shape)

        print(f'Shape X = {self.X.shape}, y = {len(self.y.shape)}')

    def predict(self, X):
        print(self.model.predict(X))

    def train_others(self, epochs = 25):          

        # Reshape X
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], -1))

        # transform y to a binary matrix using one-hot encoding
        self.y_train = y_train = np.argmax(self.y_train, axis=1)
        self.y_test = y_train = np.argmax(self.y_test, axis=1)

        # Define the preweighted models
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=epochs),
            'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=epochs),
            'SVM': SVC(class_weight='balanced', max_iter=epochs),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=epochs),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=10),
            'KNN': KNeighborsClassifier(weights='distance', n_neighbors=10),
            'Multi-layer Perceptron': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=epochs),
            'AdaBoost': AdaBoostClassifier(n_estimators=epochs)
            # 'XGBoost': XGBClassifier(n_estimators=epochs, max_depth=10, objective='multi:softmax', num_class=15)
        }

        
        score = self.plot(models, epochs, name = 'generic_models')

        return score

    def train_created_models(self, epochs = 25, k1 = 56, k2 = 56, k3 = 28):

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], self.X_train.shape[2], 1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], self.X_test.shape[2], 1))

        print(self.X_train.shape)

        # Define the CNN architecture
        cnn_model = Sequential([
            layers.Conv2D(filters=k1, kernel_size=(3, 3), activation='relu', input_shape=(14, 4, 1), padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            layers.Conv2D(filters=k2, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            layers.Flatten(),
            layers.Dense(units=k3, activation='relu'),
            layers.Dense(units=15, activation='softmax')
        ])

        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print()

        # Define the model
        dnn_model = Sequential([
            layers.Reshape(target_shape=(14, 4), input_shape=(14, 4, 1)),
            layers.Flatten(),
            Dense(56, activation='relu', input_shape=(14, 4)),
            Dropout(0.2),
            Dense(56, activation='relu'),
            Dropout(0.2),
            Dense(15, activation='softmax')
        ])

        dnn_model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])        

        # Define the RNN architecture
        rnn_model = Sequential([
            layers.Reshape(target_shape=(14, 4), input_shape=(14, 4, 1)),
            layers.SimpleRNN(units=56, return_sequences=True),
            layers.SimpleRNN(units=56),
            layers.Dense(units=15, activation='softmax')
        ])
        rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        lstm_model = Sequential()
        lstm_model.add(LSTM(56, input_shape=(14, 4), return_sequences=True))
        lstm_model.add(BatchNormalization())
        lstm_model.add(Dropout(0.8))
        lstm_model.add(LSTM(56, return_sequences=True, kernel_regularizer=L2(0.01)))
        lstm_model.add(BatchNormalization())
        lstm_model.add(Dropout(0.8))
        lstm_model.add(LSTM(28, return_sequences=False))
        lstm_model.add(Dense(15, activation='softmax'))

        # Compile the model
        lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Models
        models = {
            # "CNN" : cnn_model,
            "RNN" : rnn_model,
            # "LSTM" : lstm_model,
            # "DNN" : dnn_model
        }
        
        params, score = self.plot(models, epochs, name = 'cnn_lstm_rnn', params = cnn_model.count_params())

        return params, score

    def plot(self, models, epochs, name = '', params = 0):

        # Train and evaluate each model for X epochs
        scores = []
        times = []

        for name, model in models.items():
            start_time = time.time()

            try :
                history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs = epochs)            
                y_pred = model.predict(self.X_test)

                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(f'output_graph/Angles_detection/accuracy_{name}.png')
                plt.clf()

                score = accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(y_pred, axis=1))
                scores.append(score)
                times.append(time.time() - start_time)

                fig = plt.figure(figsize=(20, 20))
                cm = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(y_pred, axis=1))                
                row_sums = cm.sum(axis=1)
                col_sums = cm.sum(axis=0)             
                normalized_cm = cm / row_sums[:, np.newaxis]
                plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.colorbar()
                plt.xticks(range(15), [i+1 for i in range(15)], rotation=45)
                plt.yticks(range(15), [i+1 for i in range(15)])
                plt.xlabel('Predicted label')
                plt.ylabel('True label')
                plt.title(f'CM {name}')

                # Add values of each box as text
                for i in range(normalized_cm.shape[0]):
                    for j in range(normalized_cm.shape[1]):
                        plt.text(j, i, format(normalized_cm[i, j], '.2f'), ha='center', va='center')

                plt.savefig(f'output_graph/Angles_detection/cm_{name}.png', dpi=300)
                # save_model(self.model, f'PoseEstimation/fcnn_batch{batch_size}.h5')
                plt.clf()

                save_model(model, f'models/{name}.h5')

                print(f"{name}: {score}")

                # return params, score

            except :

                model.fit(self.X_train, self.y_train)            
                y_pred = model.predict(self.X_test)
                
                score = accuracy_score(self.y_test, y_pred)
                scores.append(score)
                times.append(time.time() - start_time)
                cm = confusion_matrix(self.y_test, y_pred)   
                row_sums = cm.sum(axis=1)
                col_sums = cm.sum(axis=0)             
                normalized_cm = cm / row_sums[:, np.newaxis]

                fig = plt.figure(figsize=(20, 20))
                plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.colorbar()
                plt.xticks(range(15), [i+1 for i in range(15)], rotation=45)
                plt.yticks(range(15), [i+1 for i in range(15)])
                plt.xlabel('Predicted label')
                plt.ylabel('True label')
                plt.title(f'CM {name}')

                # Add values of each box as text
                for i in range(normalized_cm.shape[0]):
                    for j in range(normalized_cm.shape[1]):
                        plt.text(j, i, format(normalized_cm[i, j], '.2f'), ha='center', va='center')

                plt.savefig(f'output_graph/Angles_detection/cm_{name}.png', dpi=300)
                save_model(model, f'models/{name}.h5')
                plt.clf()

                print(f"{name}: {score}")

                # return params, score

        if self.plot_bool :
            # Plot the results in a histogram
            fig, ax = plt.subplots()
            ax.bar(models.keys(), scores)
            ax.set_ylabel('Accuracy')
            ax.set_xticklabels(models.keys(), rotation=45, ha='right')
            plt.title(f"Accuracy Scores for {epochs} epochs")
            plt.tight_layout()
            plt.savefig(f'output_graph/Angles_detection/{name}_acc.png')
            plt.show()

            # Plot the results in a histogram
            fig, ax = plt.subplots()
            ax.bar(models.keys(), times)
            ax.set_ylabel('Time')
            ax.set_xticklabels(models.keys(), rotation=45, ha='right')
            plt.title(f"Accuracy Scores for {epochs} epochs")
            plt.tight_layout()
            plt.savefig(f'output_graph/Angles_detection/{name}_time.png')
            plt.show()


    def train_parameters(self, epochs = 50):

        # Reshape X
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], -1))

        # transform y to a binary matrix using one-hot encoding
        self.y_train = y_train = np.argmax(self.y_train, axis=1)
        self.y_test = y_train = np.argmax(self.y_test, axis=1)

        # Define parameter values to test
        lr_params = [0.001, 0.01, 0.1, 0.2, 0.25]
        rf_params = [10, 50, 100, 150, 200]
        dt_params = [2, 4, 6, 8, 10, 20, 30, 50]

        params = ['lr', 'rf', 'dt']

        # Define model dictionary with different parameter values
        models = {
            'Logistic Regression': [LogisticRegression(class_weight='balanced', max_iter=epochs, C=c) for c in lr_params],
            'Random Forest': [RandomForestClassifier(class_weight='balanced', n_estimators=n) for n in rf_params],
            'Decision Tree': [DecisionTreeClassifier(class_weight='balanced', max_depth=d) for d in dt_params],
        }

        # Train and evaluate models for each parameter value
        accuracies = {}
        for name, model_list in models.items():
            accs = []
            for i, model in enumerate(model_list):
                model.fit(self.X_train, self.y_train)
                acc = model.score(self.X_test, self.y_test)
                accs.append(acc)
            accuracies[name] = accs

        print(accuracies)

        # Plot histograms for each model and parameter
        fig, axs = plt.subplots(1, len(models), figsize=(15, 10))
        for i, (name, accs) in enumerate(accuracies.items()):
            axs[i].bar(range(len(accs)), accs)
            axs[i].set_xticks(range(len(accs)))
            if name == 'Logistic Regression':
                axs[i].set_xticklabels([str(p) for p in lr_params])
            elif name == 'Random Forest':
                axs[i].set_xticklabels([str(p) for p in rf_params])
            elif name == 'Decision Tree':
                axs[i].set_xticklabels([str(p) for p in dt_params])
            axs[i].set_xlabel(f'Parameter {params[i]}')
            axs[i].set_ylabel('Accuracy')
            axs[i].set_title(name)

        plt.tight_layout()
        plt.savefig('output_graph/Angles_detection/features/models_features.png')
        plt.show()


def test_batch():

    fcnn = FCNN_Model()

    for batch in [8, 16, 32, 64, 128, 256]:

        plt.plot(fcnn.train(epochs = 75, batch_size = batch))

    plt.title('Val accuracy for batch sizes (data augmentation)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend([str(i) for i in [8, 16, 32, 64, 128, 256]], loc='upper left')
    plt.savefig(f'PoseEstimation/images/accuracy_many_batch.png')

def three_accuracy():

    values = [0.2, 0.4, 0.6, 0.8, 1]
    score = []
    percent = []
    size = []

    for i in values:
        for j in values:

            fcnn = FCNN_Model(max_percent_change=i, max_size_change=j, plot = False)
            score.append(fcnn.train_others(epochs = 1))
            percent.append(i)
            size.append(j)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.scatter(percent, size, score)

    # Set labels for the axes
    ax.set_xlabel('max_percent_change')
    ax.set_ylabel('max_size_change')
    ax.set_zlabel('Accuracy')

    # Show the plot
    plt.show()

def add_line_to_file(arg1, arg2):
    # Open the file in append mode
    with open('test_videos/result.txt', 'a') as file:
        # Create a formatted string with the 6 arguments
        line = f"{arg1}\t{arg2}\n"
        # Write the formatted string to the file
        file.write(line)

def get_top3_values_and_indexes(arr):
    # Convert the input array to a NumPy array
    arr = np.array(arr)
    if arr.size >= 3:
        # Get the indices that would sort the array in descending order
        sorted_indices = np.argsort(arr)[::-1]
        # Extract the top 3 indices
        top3_indices = sorted_indices[:3]
        # Extract the top 3 values using the top 3 indices
        top3_values = arr[top3_indices]
        return top3_values, top3_indices



def sort_angles():

    path = "image_to_detect/angles/"
    files = [path+f for f in os.listdir(path) if f.endswith(".txt")]

    class_list = [[] for _ in range(15)]
    for file in files :
        class_list[int(file.split("_")[-2]) - 1].append(file)
    
    for cls in class_list:
        for i in range(int(len(cls)*0.2)):
            shutil.move(cls[i], 'image_to_detect/angles_test/'+ cls[i].split('/')[-1])

def accuracy_by_trainable():

    # Example data for accuracy and trainable parameters
    scores = []
    parameters = []
    k3s = []
    k2s = []

    k = [16, 32, 64, 128, 256]
    fcnn = FCNN_Model()

    # CNN - Architecure
    for k2 in k:
        for k3 in k:
            
            params, score = fcnn.train_created_models(epochs = 30, k2 = k2, k3 = k3)
            parameters.append(params)
            scores.append(score)
            k2s.append(k2)
            k3s.append(k3)

    # Create a scatter plot
    plt.scatter(parameters, scores)
    plt.xlabel('Trainable Parameters')
    plt.ylabel('Accuracy')
    plt.title('CNN Accuracy vs. Trainable Parameters (50 epochs)')
    plt.savefig('output_graph/Angles_detection/CNN_params_2d.png')
    plt.show()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data
    ax.scatter(k2s, k3s, score)

    # Set labels for the axes
    ax.set_xlabel('max_percent_change')
    ax.set_ylabel('max_size_change')
    ax.set_zlabel('Accuracy')

    # Show the plot
    plt.title('CNN Accuracy vs. CNN layers (50 epochs)')
    plt.savefig('output_graph/Angles_detection/CNN_params_3d.png')
    plt.show()



if __name__ == '__main__':

    # sort_angles()

    # Load X,y inside class definition    
    # fcnn = FCNN_Model()    
    # fcnn.train_others(epochs = 100)

    fcnn = FCNN_Model()    
    fcnn.train_created_models(epochs = 100)
    # fcnn.train_parameters(epochs = 50)
    # test_batch()

    # accuracy_by_trainable()
