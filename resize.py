import cv2
import os

TARGET_SIZE = 160

def resize_folder(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
        cv2.imwrite(path, resized)

resize_folder("dataset/real")
resize_folder("dataset/fake")

print("Resizing completed.")
