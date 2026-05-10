import os

real_count = len(os.listdir("dataset_faces_mtcnn/real"))
fake_count = len(os.listdir("dataset_faces_mtcnn/fake"))

print("Real images:", real_count)
print("Fake images:", fake_count)
