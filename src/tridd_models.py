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
        # 確保輸入 shape 與 Conv1d 相符
        self.label_comp = nn.Conv1d(label_dim, 1, kernel_size=1)
        self.label_mlp = DeepseekV3MLP(hidden_size=self.hidden_dim)
        # 將 label_feature 投射到 proj 維度
        self.label_proj = nn.Linear(self.hidden_dim, self.proj)
        
        # --- Residual Block 1 ---
        # 輸入: (B, 8, 1, 28, 28)  →  輸出: (B, 64, 1, 28, 28)
        self.conv1 = nn.Conv3d(self.proj, 16, kernel_size=(3,3,3), bias=False, padding=1)
        self.conv2 = nn.Conv3d(16, 64, kernel_size=(3,3,3), bias=False, padding=1)
        # shortcut: 將輸入從 8 通道轉成 64 通道
        self.shortcut1 = nn.Conv3d(self.proj, 64, kernel_size=(3,3,3), bias=False, padding=1)
        
        # --- Residual Block 2 ---
        # 輸入: (B, 64, 1, 28, 28) → 輸出: (B, 8, 1, 28, 28)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=(3,3,3), bias=False, padding=1)
        self.conv4 = nn.Conv3d(32, 8, kernel_size=(3,3,3), bias=False, padding=1)
        # shortcut: 將輸入從 64 通道轉成 8 通道
        self.shortcut2 = nn.Conv3d(64, 8, kernel_size=(3,3,3), bias=False, padding=1)
        
        # --- Residual Block 3 (新增) ---
        # 輸入與輸出皆為 (B, 8, 1, 28, 28)
        self.conv6 = nn.Conv3d(8, 16, kernel_size=(3,3,3), bias=False, padding=1)
        self.conv7 = nn.Conv3d(16, 8, kernel_size=(3,3,3), bias=False, padding=1)
        self.shortcut3 = nn.Identity()  # 輸入與輸出通道相同，直接用 Identity
        
        # --- Residual Block 4 (新增) ---
        # 同樣保持輸入與輸出皆為 (B, 8, 1, 28, 28)
        self.conv8 = nn.Conv3d(8, 16, kernel_size=(3,3,3), bias=False, padding=1)
        self.conv9 = nn.Conv3d(16, 8, kernel_size=(3,3,3), bias=False, padding=1)
        self.shortcut4 = nn.Identity()
        
        # --- Residual Block 5 (新增) ---
        # 同樣保持輸入與輸出皆為 (B, 8, 1, 28, 28)
        self.conv10 = nn.Conv3d(8, 16, kernel_size=(3,3,3), bias=False, padding=1)
        self.conv11 = nn.Conv3d(16, 8, kernel_size=(3,3,3), bias=False, padding=1)
        self.shortcut5 = nn.Identity()

        # --- 最後一層卷積 ---
        # 將通道從 8 轉為 1，輸出最終影像 (B, 1, 1, 28, 28)
        self.conv5 = nn.Conv3d(8, 1, kernel_size=(1,1,1), bias=False)

    def forward(self, noise, label):
        """
        noise: Tensor, shape (B, 1, 28, 28)
        label: Tensor, shape (B,) 或 (B, ) 代表標籤索引
        """
        # 1. 由 label 產生條件特徵
        # 注意：若 label 為索引，請直接傳入 Embedding 層，不需 one_hot
        # 若原本 label 為 one_hot 可先轉換為索引再使用
        label = label.long()  # 確保是整數型態
        label_emb_feature = self.label_emb(label)              # (B, hidden_dim)
        # 將 one_hot 轉換成與 Conv1d 相容的 shape：先轉換成 (B, label_dim) 再展開
        # 為此，我們先取得 one_hot 表示，再用 conv1d 處理（這裡假設需要該步驟）
        label_onehot = F.one_hot(label, num_classes=self.label_dim).float()  # (B, label_dim)
        # 將 one_hot 維度調整為 (B, label_dim, 1)
        label_onehot = label_onehot.unsqueeze(-1)
        label_comp_feature = self.label_comp(label_onehot).squeeze(1)  # (B, 1)
        # 這裡可以將 Embedding 與 label_comp_feature 作結合，或僅使用 Embedding
        label_feature = self.label_mlp(label_emb_feature)        # (B, hidden_dim)
        
        # 2. 透過線性層得到投射向量 (B, proj)
        proj_vector = self.label_proj(label_feature)             # (B, proj)
        # 調整形狀為 (B, proj, 1, 1, 1) 以便後續廣播
        proj_vector = proj_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # 3. 將 noise 擴展並用投射向量進行調製
        noise_expanded = noise.unsqueeze(1)                      # (B, 1, 1, 28, 28)
        noise_projected = noise_expanded * proj_vector           # (B, proj, 1, 28, 28)，proj=8
        
        # --- Residual Block 1 ---
        x = noise_projected                                    # (B, 8, 1, 28, 28)
        residual1 = self.shortcut1(x)                          # (B, 64, 1, 28, 28)
        out = F.relu(self.conv1(x))                            # (B, 16, 1, 28, 28)
        out = self.conv2(out)                                  # (B, 64, 1, 28, 28)
        out = F.relu(out + residual1)                          # (B, 64, 1, 28, 28)
        temp_out = out[:, -1, :, :, :]                         # (B, 1, 28, 28)
        
        # --- Residual Block 2 ---
        residual2 = self.shortcut2(out)                        # (B, 8, 1, 28, 28)
        out_block2 = F.relu(self.conv3(out))                   # (B, 32, 1, 28, 28)
        out_block2 = self.conv4(out_block2)                    # (B, 8, 1, 28, 28)
        out = F.relu(out_block2 + residual2)                   # (B, 8, 1, 28, 28)
        
        # --- Residual Block 3 (新增) ---
        residual3 = self.shortcut3(out)                        # (B, 8, 1, 28, 28)
        out_block3 = F.relu(self.conv6(out))                   # (B, 16, 1, 28, 28)
        out_block3 = self.conv7(out_block3)                    # (B, 8, 1, 28, 28)
        out = F.relu(out_block3 + residual3)                   # (B, 8, 1, 28, 28)
        
        # --- Residual Block 4 (新增) ---
        residual4 = self.shortcut4(out)                        # (B, 8, 1, 28, 28)
        out_block4 = F.relu(self.conv8(out))                   # (B, 16, 1, 28, 28)
        out_block4 = self.conv9(out_block4)                    # (B, 8, 1, 28, 28)
        out = F.relu(out_block4 + residual4)                   # (B, 8, 1, 28, 28)
        
        # --- Residual Block 5 (新增) ---
        residual5 = self.shortcut5(out)                        # (B, 8, 1, 28, 28)
        out_block5 = F.relu(self.conv10(out))                  # (B, 16, 1, 28, 28)
        out_block5 = self.conv11(out_block5)                   # (B, 8, 1, 28, 28)
        out = F.relu(out_block5 + residual5)                  # (B, 8, 1, 28, 28)

        # --- 最後一層卷積 ---
        out = self.conv5(out)                                  # (B, 1, 1, 28, 28)
        out = out.squeeze(1)                                   # 調整形狀為 (B, 1, 28, 28)
        return temp_out, out

if __name__ == "__main__":
    test_noise = torch.randn(2, 1, 28, 28)
    test_label = torch.randint(0, 10, (2,))
    model = TriDD()
    print(f"noise shape: {test_noise.shape}, label shape: {test_label.shape}")
    _,test_output = model(test_noise, test_label)
    print(test_output.shape)  # 預期為 (2, 1, 28, 28)