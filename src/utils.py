import torch

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
    for t in [1000, 500, 0]:
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

if __name__ == "__main__":
    test_x = torch.randn(2, 3, 28, 28)
    out = add_noise_process(test_x)
    print(out.shape)  # 應該是 (2, 3, 3, 28, 28)
    print(out[0, 0].shape)  # 應該是 (3, 28, 28)
    print(out[0, 1].shape)  # 應該是 (3, 28, 28)
    print(out[0, 2].shape)  # 應該是 (3, 28, 28)