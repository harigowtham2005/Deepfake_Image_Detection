from PIL import Image
import os

def clean_folder(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            img = Image.open(path)
            img.verify()
        except:
            os.remove(path)

clean_folder("dataset/real")
clean_folder("dataset/fake")

print("Dataset cleaned.")
