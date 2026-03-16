import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# ======== Video Transform Classes ========

class RandomHorizontalFlipVideo:
    """Randomly flips a video tensor [C, T, H, W] horizontally."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        # video: [C, T, H, W] tensor
        if torch.rand(1) < self.p:
            video = torch.flip(video, dims=[3])  # flip width dimension
        return video


class RandomCropVideo:
    """Randomly crops a video tensor [C, T, H, W]."""
    def __init__(self, size):
        self.crop_h, self.crop_w = size if isinstance(size, tuple) else (size, size)

    def __call__(self, video):
        # video: [C, T, H, W]
        _, _, h, w = video.shape
        if h == self.crop_h:
            top = 0
        else:
            top = torch.randint(0, h - self.crop_h + 1, (1,)).item()
        if w == self.crop_w:
            left = 0
        else:
            left = torch.randint(0, w - self.crop_w + 1, (1,)).item()
        video = video[:, :, top:top+self.crop_h, left:left+self.crop_w]
        return video


class CenterCropVideo:
    """Center crop a video tensor [C, T, H, W]."""
    def __init__(self, size):
        self.crop_h, self.crop_w = size if isinstance(size, tuple) else (size, size)

    def __call__(self, video):
        _, _, h, w = video.shape
        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        video = video[:, :, top:top+self.crop_h, left:left+self.crop_w]
        return video


# ======== FixedFramesVideoDataset ========

class FixedFramesVideoDataset(Dataset):
    def __init__(self, txt_file, target_frames=30, cache_file=None, transform=None):
        self.target_frames = target_frames
        self.cache_file = cache_file
        self.transform = transform

        # 读取样本列表
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, framerate, label = line.strip().split()
                self.samples.append({
                    'video_path': path,
                    'framerate': int(framerate),
                    'label': int(label) - 1  
                })

        # 尝试加载缓存
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file} ...", flush=True)
            cached = torch.load(cache_file)
            self.frames_list = cached['frames']
            self.labels_list = cached['labels']
            self.video_ids_list = cached['video_ids']
            print(f"[DATASET] Loaded {len(self.frames_list)} samples from cache.", flush=True)
        else:
            self.frames_list = []
            self.labels_list = []
            self.video_ids_list = []

            print("Processing videos, this may take a while ...", flush=True)
            total = len(self.samples)

            for idx, sample in enumerate(self.samples):
                video_path = sample['video_path']
                print(f"[DATASET] ({idx+1}/{total}) Loading video: {video_path}", flush=True)

                frames = self.load_video(video_path, self.target_frames)
                self.frames_list.append(frames)
                self.labels_list.append(sample['label'])
                self.video_ids_list.append(video_path)

            if cache_file:
                print(f"[DATASET] Saving dataset cache to: {cache_file}", flush=True)
                torch.save({
                    'frames': self.frames_list,
                    'labels': self.labels_list,
                    'video_ids': self.video_ids_list
                }, cache_file)
                print("[DATASET] Cache saved successfully.", flush=True)

    def __len__(self):
        return len(self.frames_list)

    def __getitem__(self, idx):
        frames = self.frames_list[idx]
        if self.transform:
            frames = self.transform(frames)  # [C, T, H, W] tensor
        label = self.labels_list[idx]
        video_id = self.video_ids_list[idx]
        return frames, torch.tensor(label, dtype=torch.long), video_id

    def load_video(self, video_path, target_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        # transpose to [C, T, H, W]
        frames = frames.transpose(3, 0, 1, 2)
        frames = torch.from_numpy(frames).float() / 255.0
        return frames

# # ======== Usage Example ========

# train_transforms = torch.nn.Sequential(
#     RandomCropVideo(224),
#     RandomHorizontalFlipVideo(0.5)
# )

# test_transforms = torch.nn.Sequential(
#     CenterCropVideo(224)
# )
