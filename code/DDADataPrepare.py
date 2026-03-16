import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FixedFramesVideoDataset(Dataset):
    def __init__(self, txt_file, target_frames=30, cache_file=None, transform=None):
        """
        Args:
            txt_file (str): 每行格式 '视频路径 帧率 类别ID'
            target_frames (int): 每段视频固定帧数
            cache_file (str or None): 存储/加载缓存的路径
        """
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

        # ====== 尝试加载缓存 ======
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file} ...", flush=True)
            cached = torch.load(cache_file, weights_only=False)
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

            print(f"[DATASET] Finished processing {total} videos.", flush=True)

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
            try:
                frames = self.transform(frames)
            except Exception as e:
                print(f"[WARN] Transform failed for video {self.video_ids_list[idx]}, using original frames")
                print(e)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(3, 0, 1, 2)
        label = self.labels_list[idx]
        video_id = self.video_ids_list[idx]
        return frames, torch.tensor(label, dtype=torch.long), video_id

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

            if h < crop_h or w < crop_w:
                # pad到最小尺寸
                pad_h = max(crop_h - h, 0)
                pad_w = max(crop_w - w, 0)
                frames = np.pad(frames, ((0,0), (pad_h//2, pad_h-pad_h//2),
                                        (pad_w//2, pad_w-pad_w//2), (0,0)), mode='constant')
                h, w = frames.shape[1], frames.shape[2]


            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            frames = frames[:, top:top+crop_h, left:left+crop_w, :]

            # frames = frames.transpose(3, 0, 1, 2)  # [C, T, H, W]
            # frames = torch.from_numpy(frames).float() / 255.0  # normalize
            return frames