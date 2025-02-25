import math
import torch
import torch.nn as nn
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

def spatial_broadcast_3d_label(z, height, width):
    """
    將輸入向量 z 進行空間廣播，並附加 x, y, z 座標資訊 (其中 z 座標固定為 0)
    輸入:
        z: Tensor, shape (B, k)   # 例如 k = hidden_dim
        height: 目標高度
        width: 目標寬度
    輸出:
        zsb: Tensor, shape (B, height, width, k+3)
    """
    B, k = z.shape
    # 1. 調整 z 的 shape 為 (B, 1, 1, k) 以便後續擴展
    z = z.view(B, 1, 1, k)
    # 2. 複製 z 到 (B, height, width, k)
    z_tiled = z.expand(B, height, width, k)
    
    # 3. 建立 x 與 y 座標，範圍從 -1 到 1
    x = torch.linspace(-1, 1, steps=width, device=z.device)
    y = torch.linspace(-1, 1, steps=height, device=z.device)
    # 注意：這裡使用 indexing='ij'，生成 y_grid 與 x_grid (shape 均為 (height, width))
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    # 4. 建立 z 座標，因為圖像為 2D，所以這裡固定為 0
    z_coord = torch.zeros((height, width), device=z.device)
    
    # 5. 將各個座標張量擴展到 (B, height, width, 1)
    x_grid = x_grid.unsqueeze(-1).expand(B, height, width, 1)
    y_grid = y_grid.unsqueeze(-1).expand(B, height, width, 1)
    z_grid = z_coord.unsqueeze(-1).expand(B, height, width, 1)
    
    # 6. 沿最後一個維度串接：得到 (B, height, width, k+3)
    zsb = torch.cat([z_tiled, x_grid, y_grid, z_grid], dim=-1)
    return zsb

# 範例模型，img 為 2D (例如 shape 為 (B, C, H, W))
class TriCD(nn.Module):
    def __init__(self, img_res=(28, 28, 1), label_dim=10):
        """
        img_res: tuple，代表圖像解析度，例如 (高度, 寬度, 通道數)
        label_dim: 標籤種類數
        """
        super().__init__()
        self.img_res = img_res
        self.label_dim = label_dim
        self.hidden_dim = img_res[0] * 8  # 例如 28*8 = 224
        
        self.label_emb = nn.Embedding(label_dim, self.hidden_dim)
        # 此處 label_comp 僅為示範，確保輸入 shape 與 Conv1d 相符
        self.label_comp = nn.Conv1d(label_dim, 1, kernel_size=1)
        self.label_mlp = DeepseekV3MLP(hidden_size=self.hidden_dim)

    def forward(self, noise, label):
        """
        noise: Tensor, shape (B, C, H, W)
        label: Tensor, shape (B, label_dim)
        """
        # 透過 embedding 將 label 轉換為特徵向量 (B, hidden_dim)
        label_emb_feature = self.label_emb(label)  # (B, hidden_dim)
        label_emb_feature = label_emb_feature.unsqueeze(1)  # (B, 1, hidden_dim)
        label_emb_feature = self.label_comp(label_emb_feature).squeeze(1)  # (B, hidden_dim)
        label_feature = self.label_mlp(label_emb_feature)  # (B, hidden_dim)
        
        B, C, H, W = noise.shape
        
        # 利用 spatial_broadcast_3d_label 將 label_feature 擴展至 3 維空間
        # 得到 shape (B, H, W, hidden_dim+3)
        label_feature_sb = spatial_broadcast_3d_label(label_feature, H, W)
        # 若後續融合需要 channel-first 格式，則轉換為 (B, hidden_dim+3, H, W)
        label_feature_sb = label_feature_sb.permute(0, 3, 1, 2)
        
        # 融合範例：直接在 channel 維度串接 noise 與 label_feature_sb
        fused = torch.cat([noise, label_feature_sb], dim=1)  # (B, C + hidden_dim+3, H, W)
        return fused

if __name__ == "__main__":
    test_noise = torch.randn(2, 1, 28, 28)
    test_label = torch.randint(0, 10, (2, 10))
    model = TriCD()
    test_output = model(test_noise, test_label)
    print(test_output.shape)  # 預期為 (2, 1+224+2, 28, 28)
    # model = TriCD()
    # print(model)
    # print(model.label_mlp)