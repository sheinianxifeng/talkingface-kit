import os
import numpy as np
import cv2
import dlib


def extract_landmarks(image_dir, output_base_dir, predictor_path="shape_predictor_68_face_landmarks.dat"):
    os.makedirs(output_base_dir, exist_ok=True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    image_count = 0  # 跟踪处理的图像数量

    for image_folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, image_folder)
        if os.path.isdir(folder_path):
            # 为每个子文件夹创建一个对应的输出目录
            subfolder_output_dir = os.path.join(output_base_dir, image_folder)
            os.makedirs(subfolder_output_dir, exist_ok=True)

            print(f"Processing folder: {image_folder}")
            for image_file in os.listdir(folder_path):
                if image_file.endswith(".jpg"):
                    image_count += 1
                    print(f"Processing image {image_count}: {image_file}")
                    image_path = os.path.join(folder_path, image_file)
                    img = cv2.imread(image_path)
                    if img is None:  # 检查图像是否加载成功
                        print(f"Failed to load image: {image_path}")
                        continue

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)

                    if len(faces) == 0:
                        print(f"No faces detected in {image_file}")

                    for face in faces:
                        landmarks = predictor(gray, face)
                        coords = np.array([[p.x, p.y] for p in landmarks.parts()])
                        output_file_path = os.path.join(subfolder_output_dir, image_file.replace(".jpg", ".lms"))

                        try:
                            # 保存 landmarks 文件
                            with open(output_file_path, 'w') as f:
                                np.savetxt(f, coords)
                                f.flush()  # 强制刷新文件写入
                            print(f"Saved landmarks for {image_file} to {output_file_path}")

                            # 检查文件是否成功保存
                            if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
                                print(f"Successfully saved landmarks: {output_file_path}")
                            else:
                                print(f"Warning: Failed to save landmarks for {image_file}")

                        except Exception as e:
                            print(f"Error saving {image_file}: {e}")

    print(f"Landmarks 提取完成，保存在 {output_base_dir}")


# 示例调用
image_directory = "./HDTF/images"
landmark_output_base_directory = "./HDTF/landmarks"
extract_landmarks(image_directory, landmark_output_base_directory)
