import cv2
import os

# INPUT VIDEO FOLDERS
REAL_VIDEO_DIR = "celeb_videos/real"
FAKE_VIDEO_DIR = "celeb_videos/fake"

# OUTPUT IMAGE FOLDERS
REAL_OUTPUT_DIR = "dataset/real"
FAKE_OUTPUT_DIR = "dataset/fake"

# SETTINGS
FRAMES_PER_VIDEO = 25   # 20–30 recommended
FRAME_SKIP = 30         # skip 30 frames each time

os.makedirs(REAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(FAKE_OUTPUT_DIR, exist_ok=True)


def extract_frames(video_dir, output_dir, label):
    video_files = os.listdir(video_dir)

    for video_name in video_files:
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Skipping {video_name}")
            continue

        frame_count = 0
        saved_count = 0

        print(f"Processing: {video_name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                filename = f"{label}_{video_name}_{saved_count}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                saved_count += 1

            if saved_count >= FRAMES_PER_VIDEO:
                break

            frame_count += 1

        cap.release()

    print(f"Finished extracting for {label}")


# Run extraction
extract_frames(REAL_VIDEO_DIR, REAL_OUTPUT_DIR, "real")
extract_frames(FAKE_VIDEO_DIR, FAKE_OUTPUT_DIR, "fake")

print("All videos processed successfully.")
