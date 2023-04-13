import glob
import pandas as pd
import os
import matplotlib.pyplot as plt

path = "dataset/"
folders = glob.glob(os.path.join(path, 'videos/**/*.mp4'), recursive=True)
print(len(folders))
df = pd.DataFrame(columns=['Gesture','Orientation', 'Time', 'Rain','Fog','Wind','Snow'])

#print(folders)
print(df)

for folder_name in folders : 

    types = folder_name.split('.')[0].split('\\')[-1].split('_')
    orientation = types[0]
    time = types[1]
    weather = types[2].split('-')
    gesture = int(types[3])
    

    rain=True if 'rain' in weather else False
    fog=True if 'fog' in weather else False
    wind=True if 'wind' in weather else False
    snow=True if 'snow' in weather else False

    list = [gesture, orientation, time, rain , fog , wind , snow]
    df = df.append(pd.DataFrame([list], 
            columns=['Gesture','Orientation', 'Time', 'Rain','Fog','Wind','Snow']), 
            ignore_index=True)

print(df.index)

for name in df.columns:
    print(name)
    df2 = df.groupby([name])[name].count()
    ax = df2.plot.bar(rot=0, color = 'blue')
    plt.title(f"Distribution for {name}")
    plt.savefig(f'output_graph/clement/{name}.png')
    plt.clf()

    