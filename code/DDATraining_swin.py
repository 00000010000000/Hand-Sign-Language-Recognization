import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from DataPrepare_improve import CachedVideoDataset
from DataPrepare_improve import preprocess_and_cache_videos

# from DDADataPrepare import FixedFramesVideoDataset  # 你之前写的 Dataset
from swin_video import SwinVideo
# from custom_models import SignLanguageRecognitionModel, I3DFeatureExtractor


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 只使用GPU1


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy


def run(train_txt, test_txt, batch_size=2, num_epochs=100, patience=5):
    # ====== 数据集 ======
    # train_dataset = FixedFramesVideoDataset(train_txt, target_frames=30)
    # print(f"Number of samples in dataset: {len(train_dataset)}")
    # test_dataset = FixedFramesVideoDataset(test_txt, target_frames=30)
    train_dataset = CachedVideoDataset('./cached_train')
    test_dataset = CachedVideoDataset('./cached_test')
    print(f"Number of samples in dataset: {len(train_dataset)}")

    # ====== 类别权重 ======
    all_labels = np.array([label.item() for _, label, _ in train_dataset], dtype=np.int64)
    class_counts = np.bincount(all_labels)
    epsilon = 1e-6
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)

    # WeightedRandomSampler
    sample_weights = class_weights[all_labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # ====== DataLoader ======
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataloaders = {'train': train_loader, 'test': val_loader}

    # ====== 模型 ======
    num_classes = len(class_counts)
    model = SwinVideo(num_classes, pretrained_safetensors_path="/home/user5/DDA4220_GroupProject/Sign-Language-Recognition/code/swinv2_tiny.safetensors")
    # model = SwinVideo(num_classes)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 明确指定只在GPU1上运行DataParallel
    model = nn.DataParallel(model, device_ids=[0])  # 由于已经设置CUDA_VISIBLE_DEVICES='1'，这里的0实际上是可见设备中的第一个，即GPU1

    # ====== 冻结 backbone 部分层 ======
    for name, param in model.module.backbone.named_parameters():
        if "layers.0" in name or "layers.1" in name:   # freeze stage1, stage2
            param.requires_grad = False

    # ====== 优化器 ======
    head_params = list(model.module.backbone.head.parameters())
    # backbone_params = [p for p in model.module.backbone.parameters() if p not in head_params]
    backbone_params = [p for p in model.module.backbone.parameters() if id(p) not in {id(hp) for hp in head_params}]

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 1e-4},
    ], weight_decay=1e-4)

    # optimizer = optim.Adam([
    #     {"params": [p for p in model.module.backbone.parameters() if p.requires_grad], "lr": 1e-5},
    #     {"params": model.module.fc.parameters(), "lr": 1e-4},  # 训练你的分类头
    # ], weight_decay=1e-4)
    # optimizer = optim.Adam([
    #     {"params": model.module.backbone.parameters(), "lr": 1e-5},
    #     {"params": model.module.backbone.head.parameters(), "lr": 1e-4},
    # ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ====== checkpoint ======
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_accuracy = 0
    early_stop = False
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # ====== Training phase ======
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        total_batches = 0
        # accumulation_steps = 6  # 🆕调整累积步数
        # optimizer.zero_grad()  # 🆕在循环外先清零梯度

        for batch_idx, (inputs, labels, vids) in enumerate(dataloaders['train']):
            inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
            # inputs = inputs.to(device)
            # inputs = inputs.cuda(non_blocking=True)
            labels = labels.to(device)

            print(f"[Train] Batch {batch_idx}: inputs.shape={inputs.shape}, labels.shape={labels.shape}")

            optimizer.zero_grad()
            outputs = model(inputs)

            # print(f"DEBUG: outputs.shape={outputs.shape}, labels.shape={labels.shape}")

            loss = criterion(outputs, labels)
            # loss = loss / accumulation_steps  # 🆕梯度平均
            loss.backward()
            optimizer.step()
            # # 每 accumulation_steps 才更新一次参数
            # if (batch_idx + 1) % accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()

            accuracy = calculate_accuracy(outputs, labels)
            # running_loss += loss.item() * accumulation_steps  # 🆕恢复原始 loss
            running_loss += loss.item()
            running_accuracy += accuracy
            total_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloaders['train'])}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

        # # 🆕处理最后剩余的 batch（如果不能整除 accumulation_steps）
        # if (batch_idx + 1) % accumulation_steps != 0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        epoch_loss = running_loss / total_batches
        epoch_accuracy = running_accuracy / total_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}, "
              f"Training Accuracy: {epoch_accuracy:.2f}%")

        # ====== Validation phase ======
        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        val_total_batches = 0

        with torch.no_grad():
            for val_inputs, val_labels, val_vids in dataloaders['test']:
                val_inputs = val_inputs.permute(0, 2, 1, 3, 4).to(device)
                # val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                # print(f"[Val] inputs.shape={val_inputs.shape}, labels.shape={val_labels.shape}")

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_accuracy = calculate_accuracy(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                val_running_accuracy += val_accuracy
                val_total_batches += 1

        val_epoch_loss = val_running_loss / val_total_batches
        val_epoch_accuracy = val_running_accuracy / val_total_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_epoch_loss:.4f}, "
              f"Validation Accuracy: {val_epoch_accuracy:.2f}%\n")

        # Scheduler step
        scheduler.step()

        # Early stopping
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            epochs_no_improve = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{epoch}_{val_epoch_accuracy:.0f}.pth")
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"Validation accuracy improved. Model saved to {checkpoint_path}\n")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).\n")
            # if epochs_no_improve >= patience:
            #     print("Early stopping triggered!")
            #     early_stop = True
            #     break

    if not early_stop:
        final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save(model.module.state_dict(), final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")


if __name__ == "__main__":
    train_txt = '/home/user5/DDA4220_GroupProject/Train_Test_Split/Class_number_90/rgb/train_class_90_rgb.txt'
    test_txt = '/home/user5/DDA4220_GroupProject/Train_Test_Split/Class_number_90/rgb/test_class_90_rgb.txt'
 
    preprocess_and_cache_videos(train_txt, './cached_train', target_frames=30)
    preprocess_and_cache_videos(test_txt, './cached_test', target_frames=30)

    run(
        train_txt='./cached_train',
        test_txt='./cached_test',
        # pretrained_i3d_weights=pretrained_weights,
        batch_size=4,
        num_epochs=10,
        patience=5
    )
   
    # batch_size = 12
    # num_epochs = 10
    # patience = 5

    # run(train_txt, test_txt, batch_size=batch_size, num_epochs=num_epochs, patience=patience)
