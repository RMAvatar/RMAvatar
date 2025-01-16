import cv2
import os

video_path = '/workspace/psen/SplattingAvatar-master/dataset/people/people_snapshot_public/male-9-plaza/male-9-plaza.mp4'
output_folder = '/workspace/psen/SplattingAvatar-master/dataset/people/peoplesnapshot/output/male-9-plaza/images'
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f'image_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames.")

