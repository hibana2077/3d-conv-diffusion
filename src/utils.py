import torch
import math
import numpy as np

def add_noise_process(x):
    """
    對輸入圖像 x 加入不同程度的噪聲，輸出形狀為 (B, 3, ch, W, H)
    三個通道分別對應於 t=1000, t=500, t=0 的噪聲程度。
    
    參數:
      x (tensor): 原始圖像，形狀為 (B, ch, W, H)
      
    返回:
      out (tensor): 加噪後的圖像，形狀為 (B, 3, ch, W, H)
    """
    B, ch, W, H = x.shape
    device = x.device
    dtype = x.dtype
    out = []
    for t in [1000, 8, 0]:
        # 計算噪聲強度比例 s (t=0: 無噪聲, t=1000: 全噪聲)
        s = t / 1000.0
        # 混合係數：sqrt(1-s) 用於原圖，sqrt(s) 用於噪聲
        sqrt_coeff_clean = torch.sqrt(torch.tensor(1-s, device=device, dtype=dtype))
        sqrt_coeff_noise = torch.sqrt(torch.tensor(s, device=device, dtype=dtype))
        # 生成與 x 相同形狀的高斯噪聲
        noise = torch.randn_like(x)
        # 生成加噪圖像
        noisy_image = sqrt_coeff_clean * x + sqrt_coeff_noise * noise
        out.append(noisy_image)
    # 將三個結果沿著新維度堆疊，得到形狀 (B, 3, ch, W, H)
    return torch.stack(out, dim=1)

def torch_cov(x: torch.Tensor) -> torch.Tensor:
    """
    使用 PyTorch 計算協方差矩陣，假設 x 的形狀為 (n, d)，其中 n 為樣本數，d 為特徵數。
    """
    n = x.size(0)
    mean = torch.mean(x, dim=0, keepdim=True)
    x_centered = x - mean
    return (x_centered.t() @ x_centered) / (n - 1)

def fid_score_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    # 將 x 與 y 攤平成 (B, features)
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)
    
    # 計算均值
    mu_x = torch.mean(x_flat, dim=0)
    mu_y = torch.mean(y_flat, dim=0)
    
    # 使用純 PyTorch 計算協方差矩陣
    sigma_x = torch_cov(x_flat)
    sigma_y = torch_cov(y_flat)
    
    # 加入小正則項防止數值不穩定
    eps = 1e-6
    device = x.device
    sigma_x = sigma_x + eps * torch.eye(sigma_x.size(0), device=device)
    sigma_y = sigma_y + eps * torch.eye(sigma_y.size(0), device=device)
    
    # 均值差的平方
    diff = mu_x - mu_y
    diff_sq = torch.dot(diff, diff)
    
    # 利用 eigen decomposition 計算 sigma_x 的矩陣平方根
    eigvals_x, eigvecs_x = torch.linalg.eigh(sigma_x)
    sqrt_sigma_x = eigvecs_x @ torch.diag(torch.sqrt(torch.clamp(eigvals_x, min=0))) @ eigvecs_x.t()
    
    # 計算中間矩陣 A = sqrt_sigma_x * sigma_y * sqrt_sigma_x
    A = sqrt_sigma_x @ sigma_y @ sqrt_sigma_x
    eigvals_A, _ = torch.linalg.eigh(A)
    sqrt_trace = torch.sum(torch.sqrt(torch.clamp(eigvals_A, min=0)))
    
    fid = diff_sq + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * sqrt_trace
    return fid.item()

if __name__ == "__main__":
    # test_x = torch.randn(2, 3, 28, 28)
    # out = add_noise_process(test_x)
    # print(out.shape)  # 應該是 (2, 3, 3, 28, 28)
    # print(out[0, 0].shape)  # 應該是 (3, 28, 28)
    # print(out[0, 1].shape)  # 應該是 (3, 28, 28)
    # print(out[0, 2].shape)  # 應該是 (3, 28, 28)
    test_x = torch.randn(2, 3, 28, 28).to('cuda')
    test_y = torch.randn(2, 3, 28, 28).to('cuda')
    fid_score = fid_score_torch(test_x, test_y)
    print(fid_score)