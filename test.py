import glob
import os


folder_path = 'image_to_detect/'
jpg_files = glob.glob(os.path.join(folder_path + 'detection_1/', '*.jpg'), recursive=True)
print(os.path.join(folder_path + 'detection_1', '/*.jpg'))
print(jpg_files)

# iterate over the files and delete them
for file_path in jpg_files:
    os.remove(file_path)