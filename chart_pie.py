import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.models import save_model, load_model
from sklearn.metrics import confusion_matrix
import itertools

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_hisogram_test():

    # Get test files
    path = "image_to_detect/angles_test/"
    files = [path+f for f in os.listdir(path) if f.endswith(".txt")]

    class_indices = [int(file.split('_')[-2]) for file in files]

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
    plt.title('Histogram of Test Classes')
    plt.savefig('output_graph/dataset_test.png')
    plt.show()

def mediapipe_pie_zeros():

    path = "image_to_detect/angles/"
    files = [path + f for f in os.listdir(path) if f.endswith(".txt")]
    df_list = []

    for file in files:
        try :
            df = pd.read_csv(file, delim_whitespace=True, header=None)
            df_list.append(df)
        except : 
            pass

    # count the number of NaN values and zeros in each dataframe
    nan_count = sum([df.isna().any().any() for df in df_list])
    zero_count = sum([df.eq(0).any().any() for df in df_list])

    # create a subplot with two pie charts for the NaN values and zeros
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].pie([nan_count, len(df_list)-nan_count], labels=['With NaN', 'Without NaN'], autopct='%1.1f%%')
    ax[0].set_title('MediaPipe detection with NaN values')
    ax[1].pie([zero_count, len(df_list)-zero_count], labels=['With 0', 'Without 0'], autopct='%1.1f%%')
    ax[1].set_title('MediaPipe detection with zeros')
    plt.savefig('output_graph/mediapipe_efficiency.png')
    plt.show()

def mediapipe_pie_ratios():

    path = "image_to_detect/angles/"
    files = [path + f for f in os.listdir(path) if f.endswith(".txt")]
    df_list = []

    for file in files:
        try :
            df = pd.read_csv(file, delim_whitespace=True, header=None)
            df_list.append(df)
        except : 
            pass

    ratios = []
    for df in df_list:
        count = (df == 0).any(axis=1).sum()
        if count != 0:
            ratio = count / df.shape[0]
            ratios.append(ratio)

    # create a subplot with two pie charts for the NaN values and zeros
    # plot a histogram of the ratios
    plt.hist(ratios)
    plt.title('Ratio of Rows with 0 Values')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')
    plt.savefig('output_graph/mediapipe_efficiency_ratio.png')
    plt.show()
    

def class_detection():

    # Load the model
    model = load_model('PoseEstimation/fcnn_batch8.h5')

    # Define the classes
    classes = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14', 'class_15']

    # Initialize dictionaries to hold true positives and false detections for each class
    true_positives = {cls: 0 for cls in classes}
    false_detections = {cls: 0 for cls in classes}

    class_target_list = []
    class_pred_list = []

    # Loop over all files in the directory
    for filename in os.listdir('image_to_detect/angles_test/'):
        
        if filename.endswith('.txt'):

            try :
                # Extract the target class from the filename
                target_class = 'class_' + filename.split('_')[3]
                class_target_list.append(int(filename.split('_')[3]))

                # Load the data from the file
                df = pd.read_csv(os.path.join('image_to_detect/angles_test/', filename), delim_whitespace=True, header=None)

                # Modify NaN values
                df = df.fillna(0)

                # Fill lines
                num_rows_to_add = 14 - df.shape[0]
                for i in range(num_rows_to_add):
                    row = [i, i+1, i+2, i+3]  # example data
                    df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
                
                data = np.array(df)

                # Reshape the data to the required input shape of the model
                data = np.reshape(data, (-1, 14, 4))

                # Make predictions on the data using the model
                predictions = model.predict(data)

                # Find the index of the class with the highest probability for each time step
                predicted_classes = np.argmax(predictions, axis=1) + 1
                class_pred_list.append(predicted_classes)

                # Calculate the number of true positives and false detections for the target class
                true_positives[target_class] += np.sum(predicted_classes == int(target_class.split('_')[1]))
                false_detections[target_class] += np.sum(predicted_classes != int(target_class.split('_')[1]))

            except :
                print('Invalid name')

    # Plot confusion matrix
    cm = confusion_matrix(class_target_list, class_pred_list)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks(np.arange(1, len(classes) + 1))
    ax.set_yticks(np.arange(1, len(classes) + 1))

    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        ax.text(j, i, format(cm_norm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black")

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Target')

    plt.savefig('output_graph/classifier_confusion_matrix.png', dpi=300)
    plt.show()

    # List
    print(true_positives)


    # Plot the histograms for true positives and false detections for each class
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35
    true_positives_vals = [true_positives[cls] for cls in classes]
    false_detections_vals = [false_detections[cls] for cls in classes]
    ax.bar(x - width/2, true_positives_vals, width, label='True Positives')
    ax.bar(x + width/2, false_detections_vals, width, label='False Detections')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Classifier absolute values')
    ax.set_ylim([0, 80])
    ax.legend()

    # # Normalize the histogram in percentages
    # for i, cls in enumerate(classes):
    #     total_samples = true_positives[cls] + false_detections[cls]
    #     ax.text(i-width/2, true_positives[cls]+5, f"{true_positives[cls]/total_samples*100:.2f} %", ha='center')
    #     ax.text(i+width/2, false_detections[cls]+5, f"{false_detections[cls]/total_samples*100:.2f} %", ha='center')

    plt.savefig("output_graph/classifier_efficiency_absolute.png")
    plt.show()

    # Plot the histograms for true positives and false detections for each class - normalized
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35

    for cls in classes :
        n = max(true_positives[cls] + false_detections[cls], 1)
        true_positives[cls] = true_positives[cls]/n
        false_detections[cls] = false_detections[cls]/n

    print(true_positives)

    true_positives_vals = [true_positives[cls] for cls in classes]
    false_detections_vals = [false_detections[cls] for cls in classes]
    ax.bar(x - width/2, true_positives_vals, width, label='True Positives')
    ax.bar(x + width/2, false_detections_vals, width, label='False Detections')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Classifier normalized values')
    ax.set_ylim([0, 1])
    ax.legend()

    # # Normalize the histogram in percentages
    # for i, cls in enumerate(classes):
    #     total_samples = true_positives[cls] + false_detections[cls]
    #     ax.text(i-width/2, true_positives[cls]+5, f"{true_positives[cls]/total_samples*100:.2f} %", ha='center')
    #     ax.text(i+width/2, false_detections[cls]+5, f"{false_detections[cls]/total_samples*100:.2f} %", ha='center')

    plt.savefig("output_graph/classifier_efficiency_normalized.png")
    plt.show()

if __name__ == "__main__" :

    mediapipe_pie_ratios()
    # plot_hisogram_test()
    # class_detection()
