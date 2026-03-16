import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FixedFramesVideoDataset(Dataset):
    def __init__(self, txt_file, target_frames=30):
        """
        Args:
            txt_file (str): 每行格式 '视频路径 帧率 类别ID'
            target_frames (int): 每段视频固定帧数
        """
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, framerate, label = line.strip().split()
                self.samples.append({
                    'video_path': path,
                    'framerate': int(framerate),
                    'label': int(label)
                })
        self.target_frames = target_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        label = sample['label']

        frames = self.load_video(video_path, self.target_frames)
        return frames, torch.tensor(label, dtype=torch.long), video_path

    def load_video(self, video_path, target_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []

        # 读取所有帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize短边到256，保持长宽比
            h, w, _ = frame.shape
            if h < w:
                new_h = 256
                new_w = int(w * 256 / h)
            else:
                new_w = 256
                new_h = int(h * 256 / w)
            frame = cv2.resize(frame, (new_w, new_h))
            frames.append(frame)
        cap.release()

        # 均匀抽取 target_frames 帧
        num_frames = len(frames)
        if num_frames < target_frames:
            while len(frames) < target_frames:
                frames.append(frames[-1])
        elif num_frames > target_frames:
            idxs = np.linspace(0, num_frames - 1, target_frames).astype(int)
            frames = [frames[i] for i in idxs]

        frames = np.stack(frames)  # [T, H, W, C]

        # CenterCrop 224
        crop_h, crop_w = 224, 224
        h, w = frames.shape[1], frames.shape[2]
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        frames = frames[:, top:top+crop_h, left:left+crop_w, :]

        frames = frames.transpose(3, 0, 1, 2)  # [C, T, H, W]
        frames = torch.from_numpy(frames).float() / 255.0  # normalize
        return frames

# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    txt_file = '/home/user5/DDA4220_GroupProject/Train_Test_Split/Class_number_90/rgb/test_class_90_rgb.txt'  # 替换为你的txt路径
    dataset = FixedFramesVideoDataset(txt_file, target_frames=30)
    dataloader = DataLoader(dataset, batch_size=1, sßhuffle=False, num_workers=2)

    # 测试读取
    for inputs, labels, video_ids in dataloader:
        print(inputs.shape)  # [B, 3, 30, 224, 224]
        print(labels)
        print(video_ids)
        break