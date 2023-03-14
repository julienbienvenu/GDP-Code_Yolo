from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
import os
import pandas as pd
from keras.models import save_model, load_model
from sklearn.model_selection import train_test_split

class FCNN_Model():

    def __init__(self, X = [], y = [], load_model = False):

        if load_model == False :

            # Define the model architecture
            model = Sequential()
            model.add(Flatten(input_shape=(16, 24)))  # Flatten the input into a 1D array
            model.add(Dense(128, activation='relu'))  # First fully connected layer with 128 neurons
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dense(15, activation='softmax')) # Output layer with 10 classes (assuming you have 10 classes to classify)

            # Compile the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.model = model

        else :

            self.load_model()

        if len(X) != 0:

            # Dataset
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

        else :

            self.load_training_data()
            # Dataset
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    def train(self, epochs=10, batch_size=25):

        # Train the model on your data
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=10, batch_size=32)
        save_model(self.model, 'PosutureClassification/fcnn.h5')

    def load_model(self):

        self.model = load_model('PosutureClassification/fcnn.h5')
        print('Model is successfully loaded')

    def load_training_data(self):

        X = []
        y = []
        
        path = "PostureClassification/"
        files = [path+f for f in os.listdir(path) if f.endswith(".txt")]

        for file in files:
            df = pd.read_csv(file)
            file.split('_')[0]
            X.append(df.values)
            y.append(int(file.split('_')[0][29:]))

        self.X = X
        self.y = y

    def predict(self, X):
        print(self.model.predict(X))

if __name__ == '__main__':
    
    fcnn = FCNN_Model()
    fcnn.train()
