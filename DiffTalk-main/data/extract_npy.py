import os
import numpy as np
import subprocess
import time

def remove_file_with_retry(file_path, retries=5, delay=1):
    """尝试删除文件，并在删除失败时重试"""
    for _ in range(retries):
        try:
            os.remove(file_path)
            print(f"文件已删除: {file_path}")
            return True
        except PermissionError:
            print(f"文件正在被占用，等待 {delay} 秒后重试...")
            time.sleep(delay)
    print(f"无法删除文件: {file_path}")
    return False

def extract_audio(video_dir, output_audio_dir):
    """从视频中提取音频，并保存为.npy文件"""
    os.makedirs(output_audio_dir, exist_ok=True)
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            output_path = os.path.join(output_audio_dir, os.path.splitext(video_file)[0] + ".wav")

            # 使用 subprocess 来确保 ffmpeg 完全执行完毕
            ffmpeg_command = [
                "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", output_path
            ]
            subprocess.run(ffmpeg_command, check=True)  # 等待ffmpeg命令执行完成

            # 使用 np.memmap 读取音频数据
            audio_data = np.memmap(output_path, dtype='h', mode='r')  # 16-bit PCM格式
            npy_output_path = os.path.join(output_audio_dir, os.path.splitext(video_file)[0] + ".npy")
            np.save(npy_output_path, audio_data)

            # 删除 np.memmap 对象，确保文件可以被删除
            del audio_data

            # 删除中间的 wav 文件
            remove_file_with_retry(output_path)  # 使用重试机制删除文件
    print(f"音频提取完成，保存在 {output_audio_dir}")

# 示例调用
video_directory = "./HDTF/video"
audio_output_directory = "./HDTF/audio_smooth"
extract_audio(video_directory, audio_output_directory)
