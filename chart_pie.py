import pandas as pd
import matplotlib.pyplot as plt
import os

def mediapipe_pie():

    path = "image_to_detect/angles/"
    files = [path+f for f in os.listdir(path) if f.endswith(".txt")]
    df_list = []

    for file in files:
        df = pd.read_csv(file, delim_whitespace=True, header=None)
        df_list.append(df)

    # count the number of NaN values and zeros in each dataframe
    nan_count = sum([df.isna().any().any() for df in df_list])
    zero_count = sum([df.eq(0).any().any() for df in df_list])

    # create a subplot with two pie charts for the NaN values and zeros
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].pie([nan_count, len(df_list)-nan_count], labels=['With NaN', 'Without NaN'], autopct='%1.1f%%')
    ax[0].set_title('MediaPipe detection with NaN values')
    ax[1].pie([zero_count, len(df_list)-zero_count], labels=['With 0', 'Without 0'], autopct='%1.1f%%')
    ax[1].set_title('MediaPipe detection with zeros')
    plt.show()

if __name__ == '__main__':
    mediapipe_pie()
