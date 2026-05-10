import os
import cv2
from mtcnn import MTCNN

INPUT_DATASET = "dataset"
OUTPUT_DATASET = "dataset_faces_mtcnn"

os.makedirs(os.path.join(OUTPUT_DATASET, "real"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DATASET, "fake"), exist_ok=True)

detector = MTCNN()

def process_folder(label):
    input_folder = os.path.join(INPUT_DATASET, label)
    output_folder = os.path.join(OUTPUT_DATASET, label)

    for file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_img)

            if len(faces) == 0:
                continue

            x, y, w, h = faces[0]['box']

            # Fix negative coordinates
            x = max(0, x)
            y = max(0, y)

            face = img[y:y+h, x:x+w]

            # Skip very small crops
            if face.shape[0] < 40 or face.shape[1] < 40:
                continue

            face_resized = cv2.resize(face, (224, 224))
            cv2.imwrite(os.path.join(output_folder, file), face_resized)

        except:
            continue

    print(f"{label} done.")

process_folder("real")
process_folder("fake")

print("All images processed safely.")
