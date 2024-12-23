import cv2
import os

def extract_images(video_dir, output_image_dir, fps=25):
    os.makedirs(output_image_dir, exist_ok=True)
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            output_path = os.path.join(output_image_dir, os.path.splitext(video_file)[0])
            os.makedirs(output_path, exist_ok=True)

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 每隔 frame_rate/fps 帧保存一个图像
                if int(cap.get(1)) % (frame_rate // fps) == 0:
                    frame_name = os.path.join(output_path, f"{count:05d}.jpg")
                    cv2.imwrite(frame_name, frame)
                    count += 1

            cap.release()
    print(f"图片提取完成，保存在 {output_image_dir}")

# 示例调用
video_directory = "./HDTF/video"
image_output_directory = "./HDTF/images"
extract_images(video_directory, image_output_directory)
