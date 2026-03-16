import os
import cv2
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def preprocess_and_cache_videos(txt_file, cache_dir, target_frames=30, resize=(256, 256)):
    """
    将视频预处理后保存为 tensor 文件 (.pt)
    每个视频只处理一次，训练时直接加载 tensor
    """
    os.makedirs(cache_dir, exist_ok=True)

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        path, framerate, label = line.strip().split()
        video_name = os.path.basename(path).replace('.mp4','')
        cache_file = os.path.join(cache_dir, f"{video_name}.pt")

        if os.path.exists(cache_file):
            continue  # 已经缓存过

        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total-1, target_frames).astype(int)
        idx_set = set(frame_indices)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count in idx_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, resize)
                frames.append(frame)
            count += 1
        cap.release()

        if len(frames) < target_frames:
            while len(frames) < target_frames:
                frames.append(frames[-1])

        frames = np.stack(frames)  # (T, H, W, C)
        # 转成 [C, T, H, W] 直接兼容原训练脚本
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
        # frames = torch.from_numpy(frames).transpose(3,0,1,2).float() / 255.0
        label = torch.tensor(int(label))

        torch.save((frames, label), cache_file)
        print(f"Cached {video_name} -> {cache_file}")


class CachedVideoDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        frames, label = torch.load(self.files[idx])
        return frames, label, os.path.basename(self.files[idx])


if __name__ == "__main__":
    # train_txt = '/home/user5/DDA4220_GroupProject/Train_Test_Split/Class_number_90/rgb/train_class_90_rgb.txt'
    # test_txt  = '/home/user5/DDA4220_GroupProject/Train_Test_Split/Class_number_90/rgb/test_class_90_rgb.txt'

    # # 预处理并缓存视频（只需运行一次）
    # preprocess_and_cache_videos(train_txt, "./cached_train", target_frames=30)
    # preprocess_and_cache_videos(test_txt, "./cached_test", target_frames=30)

    train_dataset = CachedVideoDataset("./cached_train")
    val_dataset   = CachedVideoDataset("./cached_test")

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=2)

    # 测试读取
    for inputs, labels, video_ids in train_loader:
        print(inputs.shape)  # [B, C, T, H, W]
        print(labels)
        print(video_ids)
        break
