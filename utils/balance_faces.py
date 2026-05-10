import os
import random

real_path = "dataset_faces_mtcnn/real"
fake_path = "dataset_faces_mtcnn/fake"

real_images = os.listdir(real_path)
fake_images = os.listdir(fake_path)

real_count = len(real_images)
fake_count = len(fake_images)

print("Before balancing:")
print("Real:", real_count)
print("Fake:", fake_count)

if fake_count > real_count:
    extra_fake = random.sample(fake_images, fake_count - real_count)

    for file in extra_fake:
        os.remove(os.path.join(fake_path, file))

print("After balancing:")
print("Real:", len(os.listdir(real_path)))
print("Fake:", len(os.listdir(fake_path)))
