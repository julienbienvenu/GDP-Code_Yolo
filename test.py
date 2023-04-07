# Generate the folders
import glob
import os
import numpy as np

from matplotlib import pyplot as plt

def histo():

    path = "dataset/"
    folders = glob.glob(os.path.join(path, '**/*.mp4'), recursive=True)
    class_indices = []

    for file in folders:
        class_indices.append(int(file.split('\\')[-1].split('.')[0].split('_')[-2]))

    class_counts = {}
    for label in class_indices:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    n_bins = 15
    print(class_counts)
    print(len(class_counts))
    hist, bins = np.histogram(class_indices, bins=n_bins)
    plt.hist(class_indices, bins=n_bins)
    # plt.xticks(bins + 0.5*(bins[1]-bins[0]), [i+1 for i in range(n_bins)])
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Histogram of Classes in Dataset')
    plt.savefig('output_graph/dataset.png')
    plt.show()

if __name__ == '__main__':
    histo()