# DiffTalk



The pytorch implementation for our CVPR2023 paper "DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation".

[[Project\]](https://sstzal.github.io/DiffTalk/) [[Paper\]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiffTalk_Crafting_Diffusion_Models_for_Generalized_Audio-Driven_Portraits_Animation_CVPR_2023_paper.pdf) [[Video Demo\]](https://youtu.be/tup5kbsOJXc)

## Requirements



- python 3.7.0
- pytorch 1.10.0
- pytorch-lightning 1.2.5
- torchvision 0.11.0
- pytorch-lightning==1.2.5

For more details, please refer to the `requirements.txt`. We conduct the experiments with 8 NVIDIA 3090Ti GPUs.

Put the first stage model:https://ommer-lab.com/files/latent-diffusion/vq-f4.zipto `./models`.

## Dataset



Please download the HDTF dataset for training and test, and process the dataset as following.

**Data Preprocessing:**

1. Set all videos to 25 fps.
2. Extract the audio signals and facial landmarks.
3. Put the processed data in `./data/HDTF`, and construct the data directory as following.
4. Constract the `data_train.txt` and `data_test.txt` as following.

./data/HDTF:

```
|——data/HDTF
   |——images
      |——0_0.jpg
      |——0_1.jpg
      |——...
      |——N_M.bin
   |——landmarks
      |——0_0.lmd
      |——0_1.lmd
      |——...
      |——N_M.lmd
   |——audio_smooth
      |——0_0.npy
      |——0_1.npy
      |——...
      |——N_M.npy
```



./data/data_train(test).txt:

```
0_0
0_1
0_2
...
N_M
```

N是类别编号，M是序号

## Training

```
sh run.sh
```

## Test

```
sh inference.sh
```



## 数据集

下载HDTF数据集，详情见https://github.com/universome/HDTF/tree/main

下载[shape_predictor_68_face_landmarks.dat](https://github.com/yxdydgithub/difftalk_preprocess/blob/main/shape_predictor_68_face_landmarks.dat)

进入process目录处理数据：

```
1) 视频帧率转为25fps: sh preprocess/0_change_fps.sh 
2) 提取每一帧图像（images），及对应的音频（audio_wav）：python3 preprocess/1_extract_frame_audio.py 
3）使用deepspeech从.wav提取特征，保存为npy（audio_ds）：python3 preprocess/audio/deepspeech_features/extract_ds_features.py --input data/HDTF/audio_wav --output data/HDTF/audio_smooth 
	注意需要修改deepspeech_features.py中的target_sample_rate= 采样率，video_fps根据需要调整，显示维度信息（1, 16, 29）, 
	注：提示frame length (550) is greater than FFT size (512), frame will be truncated, 目前没发现对结果有影响。
4) 检测人脸关键点并归一化（landmarks）：python3 preprocess/2_detect_face_lmk.py 
5) audio_ds相邻8帧取平均，例如当前是第10帧，取7，8，9，10，11，12, 13, 14的均值，重新写入第10帧（audio_smooth）. 一般开头和结尾帧值为空，保留使用0填充: python3 preprocess/3_smooth_audio.py 
6) 使用clean_common_files.py，执行def cleanup_common_files和  filter_txt_with_folder_files('data/HDTF/images', 'data/org_data_test.txt', 'data/data_test.txt')
```



## 测试与评估

​            **1.     在DiffTalk-main项目下运行命令./inference.sh（过程较长，一个视频的所有图片大概需要半个小时，且需要手动停止，否则会直接推理至测试集结束）。图片会保存在DiffTalk-main/logs/inference。每批次处理20个，因此命名格式为批次_序号。**

​            **2.     将得到的inference图像（最好是从视频的第一帧到最后一帧，且只保留一个视频，减少后续调整）保存到DiffTalk-main/image_assessments/inference_img下，再通过测试集找到对应的原图像（在DiffTalk-main/data/HDTF/images内），并将原图像保存在DiffTalk-main/image_assessments/org_img下。**

​            **3.     执行python rename.py，得到重命名后的推理图像，该图像保存在test_img下，命名格式与原图像命名格式相同（如果不同需要在代码中手动修改保存路径命名部分）**

​            **4.     指标检测：**

​	**1）PSNR和SSIM:执行python PSNR&SSIM.py,输出PSNR&SSIM results.txt,内含各行对比结果，最后打印平均	指标。**

​	**2）NIQE:执行python niqe.py，打印出平均指标。**

​	**3）FID：执行python -m pytorch_fid org_img test_img，打印出平均指标。**

​	**4）LSE-C和LSE-D：执行python LSE-C&LSE-D.py，打印二者平均指标。**

​            **5.     视频合成：**

​            **1)     执行ffmpeg -i org_video/视频名.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 test_audio/输出音频		名.wav；**

​            **2)     执行ffmpeg -framerate 25 -i test_img/类别_%d.jpg -c:v libx264 -pix_fmt yuv420p test_video/输出视频	名.mp4；**

​            **3)     执行ffmpeg -i test_video/输出视频名.mp4 -i test_audio/输出音频名.wav -c:v copy -c:a aac -strict 	experimental result_video/输出视频名.mp4**

