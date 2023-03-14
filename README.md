# Marshall detection

## Project structure
- Dataset
    - Class01
        - c01_v01_video_name.{avi, mp4}
        - c01_v01_video_name.txt(xmin,ymin,xma,ymax)
    - Class02
    - Class03
    - ClasstoName.txt
- Yolov7
    - data
        - train
        - test
    - train.py
- main.py
- others.py

The name for each file in the dataset can be modify with a Python function.

## Database

### MIT Database
http://groups.csail.mit.edu/mug/natops/archive/
- 10 classes
- 30 videos per classes (front + different quality + same zoom)
- 650 000 img
- Not labelized (as DarkNet labelization)

### Data creation
- generic labelization
- train/val creation

## Crescent
Nightmare
- .sub file ?

## Model
We create a 3 step model :

### Yolov7
Based on class classification and labelization
Use the following classes :

| Class | Name |
| -------- | -------- |
| 0 | Marshall |
| 1 | Worker |
| 2 | Aircraft Front |
| 3 | Aircraft Back |

The objective of this algorithm is to detect the marshalls.

### PoseEstimation
Ask Bouthaina

### PostureClassification
Based on a F-CNN model (not pre-weighted), design not already defined :
- Input : Matrix PoseEstimationPoints * length (normalized by human size)
- Output : Classfication (softmax)

## Limitations
