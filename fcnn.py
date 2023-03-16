from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
import os
import pandas as pd
from keras.models import save_model, load_model
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from keras.regularizers import L2

import numpy as np

class FCNN_Model():

    def __init__(self, X = [], y = [], load_model = False):

        if load_model == False :

            model = Sequential()
            model.add(Flatten(input_shape=(32, 96)))
            model.add(Dense(16, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(8, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(8, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(15, activation='softmax'))

            # model = Sequential()
            # model.add(LSTM(32, input_shape=(32, 96), return_sequences=True)) # add LSTM layer with 32 units
            # model.add(BatchNormalization())
            # model.add(Dropout(0.8))
            # model.add(LSTM(8, return_sequences=True)) # add another LSTM layer with 16 units
            # model.add(BatchNormalization())
            # model.add(Dropout(0.8))
            # model.add(LSTM(8, return_sequences=False))
            # model.add(Dense(15, activation='softmax'))

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
            # self.data_augmentation()

    def train(self, epochs=200, batch_size=32):

        # Train the model on your data
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=epochs, batch_size=batch_size)
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('PoseEstimation/accuracy.png')

        save_model(self.model, 'PoseEstimation/fcnn.h5')

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

    def data_augmentation(self):

        print(f'Data augmentation : start / X : {self.X_train.shape}, y : {self.y_train.shape}')

        max_percent_change = 0.25
        max_size_change = 0.8
        new_x_list = []
        new_y_list = []

        for pos in range(len(self.X_train)):
            # Add the original matrix to the new list
            arr = self.X_train[pos]
            new_x_list.append(arr)
            new_y_list.append(self.y_train[pos])

            # Create 4 more matrices with 10% maximum variation
            for i in range(50):
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
                new_y_list.append(self.y_train[pos])

        self.X_train = np.array(new_x_list)
        self.y_train = np.array(new_y_list)

        print(f'Data augmentation : end / X : {self.X_train.shape}, y : {self.y_train.shape}')


    def load_model(self):

        self.model = load_model('PoseEstimation/fcnn.h5')
        print('Model is successfully loaded')

    def load_training_data(self):

        print(f'Collecting data')

        X = []
        y = []
        
        path = "PoseEstimation/txt_files/"
        files = [path+f for f in os.listdir(path) if f.endswith(".txt")]

        for file in files:
            y_p = [0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            df = pd.read_csv(file, delim_whitespace=True)
            df = df.fillna(0)
            X.append(df.values)
            y_p[int(file.split('_')[-2]) - 1] = 1
            y.append(y_p)

        self.X = np.array(X)
        self.y = np.array(y)

        print(f'Shape X = {self.X.shape}, y = {len(y)}')

    def predict(self, X):
        print(self.model.predict(X))

if __name__ == '__main__':

    # Load X,y inside class definition    
    fcnn = FCNN_Model()
    fcnn.train(epochs = 200, batch_size = 64)
