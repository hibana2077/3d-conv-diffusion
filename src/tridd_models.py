import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size=2048, intermediate_size=2048, hidden_act="gelu_fast"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# 定義一個基本的 Residual Block，用來取代原本重複的 Block 3、4、5
class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=(kernel_size, kernel_size, kernel_size),
                               bias=False, padding=padding)
        self.gn1 = nn.GroupNorm(8, mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size),
                               bias=False, padding=padding)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Identity()  # 輸入輸出通道相同，直接 Identity

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.silu(self.conv1(x))
        out = self.gn1(out)
        out = self.conv2(out)
        out = self.gn2(out)
        return F.silu(out + residual)

class TriDD(nn.Module):
    def __init__(self, img_res=(28, 28, 1), label_dim=10, proj=8):
        """
        img_res: tuple，代表圖像解析度，例如 (高度, 寬度, 通道數)
        label_dim: 標籤種類數
        proj: 希望擴張到的維度數，例如 8
        """
        super().__init__()
        self.img_res = img_res
        self.label_dim = label_dim
        self.hidden_dim = img_res[0] * 8  # 例如 28*8 = 224
        self.proj = proj

        self.label_emb = nn.Embedding(label_dim, self.hidden_dim)
        self.label_mlp = DeepseekV3MLP(hidden_size=self.hidden_dim)
        self.label_proj = nn.Linear(self.hidden_dim, self.proj)

        # --- Residual Block 1 (保留原始結構) ---
        self.conv1 = nn.Conv3d(self.proj, 16, kernel_size=(7,7,7), bias=False, padding=3)
        self.gn1 = nn.GroupNorm(8, 16)
        self.conv2 = nn.Conv3d(16, 64, kernel_size=(3,3,3), bias=False, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.shortcut1 = nn.Conv3d(self.proj, 64, kernel_size=(1,1,1), bias=False)

        # --- Residual Block 2 (保留原始結構) ---
        self.conv3 = nn.Conv3d(64, 32, kernel_size=(3,3,3), bias=False, padding=1)
        self.gn3 = nn.GroupNorm(8, 32)
        self.conv4 = nn.Conv3d(32, 8, kernel_size=(3,3,3), bias=False, padding=1)
        self.gn4 = nn.GroupNorm(8, 8)
        self.shortcut2 = nn.Conv3d(64, 8, kernel_size=(3,3,3), bias=False, padding=1)

        # --- Residual Block 3,4,5 (重複部分，用 for loop 建立) ---
        self.residual_blocks = nn.ModuleList([BasicBlock(8, 16, 8) for _ in range(10)])

        # --- 最後一層卷積 ---
        self.conv5 = nn.Conv3d(8, 1, kernel_size=(3,3,3), bias=False, padding=1)

    def forward(self, noise, label):
        """
        noise: Tensor, shape (B, 1, 28, 28)
        label: Tensor, shape (B,) 代表標籤索引
        """
        # 1. 由 label 產生條件特徵
        label = label.long()
        label_emb_feature = self.label_emb(label)  # (B, hidden_dim)
        label_feature = self.label_mlp(label_emb_feature)  # (B, hidden_dim)

        # 2. 透過線性層得到投射向量 (B, proj)
        proj_vector = self.label_proj(label_feature)
        proj_vector = proj_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, proj, 1, 1, 1)

        # 3. 將 noise 擴展並用投射向量進行調製
        noise_expanded = noise.unsqueeze(1)  # (B, 1, 1, 28, 28)
        noise_projected = noise_expanded * proj_vector  # (B, proj, 1, 28, 28)

        # --- Residual Block 1 ---
        x = noise_projected  # (B, 8, 1, 28, 28)
        residual1 = self.shortcut1(x)  # (B, 64, 1, 28, 28)
        out = F.silu(self.conv1(x))  # (B, 16, 1, 28, 28)
        out = self.gn1(out)
        out = self.conv2(out)  # (B, 64, 1, 28, 28)
        out = self.gn2(out)
        out = F.silu(out + residual1)  # (B, 64, 1, 28, 28)

        # --- Residual Block 2 ---
        residual2 = self.shortcut2(out)  # (B, 8, 1, 28, 28)
        out_block2 = F.silu(self.conv3(out))  # (B, 32, 1, 28, 28)
        out_block2 = self.gn3(out_block2)
        out_block2 = self.conv4(out_block2)  # (B, 8, 1, 28, 28)
        out_block2 = self.gn4(out_block2)
        out = F.silu(out_block2 + residual2)  # (B, 8, 1, 28, 28)
        temp_out1 = out[:, -1, :, :, :]

        # --- Residual Block 3,4,5 使用 for 迴圈 ---
        for block in self.residual_blocks:
            out = block(out)  # 每次輸出皆為 (B, 8, 1, 28, 28)
        temp_out2 = out[:, -1, :, :, :]

        # --- 最後一層卷積 ---
        out = self.conv5(out)  # (B, 1, 1, 28, 28)
        out = out.squeeze(1)   # 調整形狀為 (B, 1, 28, 28)
        return [temp_out1, temp_out2], out

if __name__ == "__main__":
    test_noise = torch.randn(2, 1, 28, 28)
    test_label = torch.randint(0, 10, (2,))
    model = TriDD()
    print(f"noise shape: {test_noise.shape}, label shape: {test_label.shape}")
    _,test_output = model(test_noise, test_label)
    print(test_output.shape)  # 預期為 (2, 1, 28, 28)