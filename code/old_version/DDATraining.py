import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from DDADataPrepare import FixedFramesVideoDataset  # 你之前写的 Dataset
from pytorch_i3d import InceptionI3d
from custom_models import SignLanguageRecognitionModel, I3DFeatureExtractor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(torch.cuda.device_count())))

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy

def run(train_txt, test_txt, pretrained_i3d_weights, batch_size=2, num_epochs=100, patience=5):
    # ====== 数据集 ======
    train_dataset = FixedFramesVideoDataset(train_txt, target_frames=30)
    print(f"Number of samples in dataset: {len(train_dataset)}")
    # print("\n=== Checking first 20 samples from train_dataset ===")
    # labels_debug = []
    # for i in range(20):
    #     try:
    #         frames, lbl, vid = train_dataset[i]
    #         print(f"Sample {i} -> label={lbl}, type={type(lbl)}, vid={vid}")
    #         labels_debug.append(lbl)
    #     except Exception as e:
    #         print(f"Error at dataset[{i}]: {e}")

    # print("Collected labels:", labels_debug)
    # print("Number of collected labels:", len(labels_debug))
    test_dataset = FixedFramesVideoDataset(test_txt, target_frames=30)

    print("Number of samples:", len(train_dataset))
    for i in range(len(train_dataset)):  # 查看所有样本
        frames, label, vid = train_dataset[i]
        print(f"Sample {i}: label={label}, type={type(label)}, video_id={vid}")

    # ====== 类别权重 ======
    all_labels = np.array([label.item() if isinstance(label, torch.Tensor) else label
                       for _, label, _ in train_dataset], dtype=np.int64)
    unique_labels = np.unique(all_labels)
    print("Unique labels:", unique_labels)
    # 创建旧标签 -> 新连续索引的映射
    label_map = {old: new for new, old in enumerate(unique_labels)}
    print("Label mapping (old -> new):", label_map)
    all_labels_mapped = np.array([label_map[l] for l in all_labels], dtype=np.int64)
    class_counts = np.bincount(all_labels_mapped)
    print("Number of classes:", len(class_counts))
    print("Samples per class:", class_counts)
    epsilon = 1e-6
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)

    # WeightedRandomSampler
    sample_weights = class_weights[all_labels_mapped]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # ====== DataLoader ======
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    dataloaders = {'train': train_loader, 'test': val_loader}

    # ====== 模型 ======
    num_classes = len(class_counts)
    i3d = InceptionI3d(100, in_channels=3)
    i3d.load_state_dict(torch.load(pretrained_i3d_weights, weights_only=True))
    feature_extractor = I3DFeatureExtractor(i3d)
    model = SignLanguageRecognitionModel(feature_extractor, num_classes)

    # 冻结 I3D 特征提取器
    for param in model.feature_extractor.feature_extractor.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model = nn.DataParallel(model)

    lr = 1e-4
    weight_decay = 1e-5  

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.module.transformer.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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

        for batch_idx, (inputs, labels, vids) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = calculate_accuracy(outputs, labels)
            running_loss += loss.item()
            running_accuracy += accuracy
            total_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloaders['train'])}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

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
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

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
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation accuracy improved. Model saved to {checkpoint_path}\n")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).\n")
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                early_stop = True
                break

    if not early_stop:
        final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}")

if __name__ == "__main__":
    train_txt = '/home/user5/DDA4220_GroupProject/Train_Test_Split/Class_number_90/rgb/train_class_90_rgb.txt'
    with open(train_txt, 'r') as f:
        lines = f.readlines()
    print(f"Number of training samples: {len(lines)}")
    print("First 5 lines:", lines[:5])

    test_txt = '/home/user5/DDA4220_GroupProject/Train_Test_Split/Class_number_90/rgb/test_class_90_rgb.txt'
    pretrained_weights = 'i3d_pretrained_100.pt'
    batch_size = 12
    num_epochs = 10
    patience = 5

    run(train_txt, test_txt, pretrained_weights, batch_size=batch_size, num_epochs=num_epochs, patience=patience)
