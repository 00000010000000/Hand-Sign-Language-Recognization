import matplotlib.pyplot as plt
import numpy as np
import cv2
from DDADataPrepare import FixedFramesVideoDataset

# -------------------------------
# 1. 加载数据集
# -------------------------------
dataset = FixedFramesVideoDataset(
    "/home/user5/DDA4220_GroupProject/Train_Test_Split/Class_number_90/rgb/train_small.txt",
    target_frames=30
)

# -------------------------------
# 2. 取第一个视频
# -------------------------------
frames, label, video_id = dataset[0]  # frames shape: [C, T, H, W]

# 转回 [T, H, W, C] 用于可视化/保存
frames_np = frames.permute(1, 2, 3, 0).numpy()  # [T, H, W, C]

# -------------------------------
# 3. 保存为 MP4
# -------------------------------
# 获取帧大小
height, width = frames_np.shape[1], frames_np.shape[2]

# 定义视频写入器
video_path = "sample_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 编码
fps = 10  # 帧率，可根据需要修改
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# 写入每一帧
for frame in frames_np:
    # 转为 uint8 并转换为 BGR（OpenCV 使用 BGR）
    frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

out.release()
print(f"Saved processed video to {video_path}")

# -------------------------------
# 4. 可视化前 5 帧
# -------------------------------
for i in range(min(5, frames_np.shape[0])):
    plt.imshow(frames_np[i])
    plt.title(f"Frame {i}")
    plt.axis('off')
    plt.show()
