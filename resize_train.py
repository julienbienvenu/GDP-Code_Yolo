import os
import random
import shutil

def send_to_yolo():
    dst_train = "yolov7/data/train_resize/"
    dst_test = "yolov7/data/val_resize/"

    image_files = ["yolov7/data/train/"+f for f in os.listdir("yolov7/data/train") if f.endswith(".jpg")]
    random.shuffle(image_files)

    train_list = image_files[:int(0.02*len(image_files))]
    test_list = image_files[int(0.995*len(image_files)):]

    for img in test_list:
        shutil.move(img, dst_test + img[18:])
        shutil.move("yolov7/data/train/" + img[18:-4]+".txt", dst_test + img[18:-4]+".txt")

    print("0.25 achieve")

    for img in train_list:
        shutil.move(img, dst_train + img[18:])
        shutil.move("yolov7/data/train/" + img[18:-4]+".txt", dst_train + img[18:-4]+".txt")    

    print('Done')

if __name__=='__main__':
    img="yolov7/data/train/test.jpg"
    print(img[18:-4])
    send_to_yolo()