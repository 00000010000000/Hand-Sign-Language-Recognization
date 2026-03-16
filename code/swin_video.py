import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

class SwinVideo(nn.Module):
    def __init__(self, num_classes, pretrained_safetensors_path=None):
        super().__init__()

        # timm 默认不加载预训练（服务器无法联网）
        self.backbone = timm.create_model(
            'swinv2_tiny_window16_256',
            pretrained=False,   # 不能从外网下载，因此必须 False
            num_classes=0
        )

        # 如果传入了本地 safetensors，则加载它
        if pretrained_safetensors_path is not None:
            print(f"Loading pretrained weights from {pretrained_safetensors_path}")
            state_dict = load_file(pretrained_safetensors_path)
            # 过滤掉head层的权重
            state_dict = {k:v for k,v in state_dict.items() if not k.startswith('head.')}
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        # 创建自己的分类头，不依赖backbone的head
        self.num_classes = num_classes
        self.classifier = None  # 将在forward中动态创建

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # 合并 batch 和时间维度
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # 提取特征
        features = self.backbone.forward_features(x)
        
        # 重要：添加全局平均池化来减小特征维度
        # 对于Swin Transformer，features的形状通常是 [B*T, H*W, C]
        # 我们需要对空间维度进行池化
        if len(features.shape) == 3:  # [B*T, num_patches, hidden_dim]
            # 对patch维度进行平均池化
            features = torch.mean(features, dim=1)
        elif len(features.shape) == 4:  # 如果是 [B*T, C, H, W] 格式
            # 对空间维度进行平均池化
            features = torch.mean(features, dim=[2, 3])
        
        # 打印特征维度以便调试
        # print(f"After pooling: features.shape = {features.shape}")
        
        # 时间平均池化
        features = features.reshape(B, T, -1).mean(dim=1)
        
        # 动态创建分类头以匹配实际特征维度
        if self.classifier is None:
            self.classifier = nn.Linear(features.shape[-1], self.num_classes).to(features.device)
            print(f"Created classifier with input dim: {features.shape[-1]}, output dim: {self.num_classes}")
        
        # 应用分类头
        outputs = self.classifier(features)
        return outputs
