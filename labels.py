import os
import xml.etree.ElementTree as ET
from frame import Frame
import random

def convert_xml_to_txt():

    # Remove the previous values
    folder_paths = ["yolov7/data/train/", "yolov7/data/val/"]  # relative path to the folder
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    cpt = 0
    good = 0
    path = 'dataset/labels/'
    folders = [f2.path for f1 in os.scandir(path) for f2 in os.scandir(f1.path)]
    # files = [f3.path for f1 in os.scandir(path) for f2 in os.scandir(f1.path) for f3 in os.scandir(f2.path) if f3.name.endswith(".xml")]

    for f2 in folders :
        files = [f3.path for f3 in os.scandir(f2) if f3.name.endswith(".xml")]        
        random_allocation = [False if random.randint(0,4)==1 else True for _ in range(len(files))]

        for i in range(len(files)):
            file = files[i]
            try :
                good += 1
                tree = ET.parse(file)
                root = tree.getroot()

                #Process the video conditions
                filemane = root.find('filename').text
                fileroot = file
                types = filemane.split('_')

                orientation = types[0]
                time = types[1]
                weather = types[2].split('-')

                rain=True if 'rain' in weather else False
                fog=True if 'fog' in weather else False
                wind=True if 'wind' in weather else False
                snow=True if 'snow' in weather else False

                path_final = 'train' if random_allocation[i] else 'val'

                for obj in root.findall('object'):

                    # We load the data as a Frame class
                    frame = Frame(
                        filename=filemane,
                        fileroot = file,
                        directory='None',
                        marshall_signal=int(types[3]),
                        orientation='front',
                        rain=rain,
                        wind=wind,
                        snow=snow,
                        fog=fog,
                        time = 'day',
                        yolo_name = obj.find('name').text,
                        x1=int(obj.find('bndbox').find('xmin').text),
                        y1=int(obj.find('bndbox').find('ymin').text),
                        x2=int(obj.find('bndbox').find('xmax').text),
                        y2=int(obj.find('bndbox').find('ymax').text),
                        img_width=int(root.find('size').find('width').text),
                        img_height=int(root.find('size').find('height').text)
                        )
                    
                    #frame.show()
                    frame.write_txt_label(path_final = path_final, marshall_classification=True)
                    

            except Exception as e:
                # print(f'XML Load failed : {file}')
                cpt += 1
                with open('dataset/Missing_labels.txt', 'a') as file:
                    file.write(str(e) +'\n')
                    file.close()

                print(e)
    
    print(f'Number of errors : {cpt}, so {round(cpt/(cpt+good),2)}')
    return 0

if __name__ == '__main__': 

    convert_xml_to_txt()
    