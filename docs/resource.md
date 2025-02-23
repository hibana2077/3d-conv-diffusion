# Resource

## Survey

- [Image-Fusion](https://github.com/Linfeng-Tang/Image-Fusion)
- [ivif_zoo](https://github.com/rollingplain/ivif_zoo)

## Dataset

Dataset | Fusion Scenario (FS) | Img Pairs | Resolution | Color | Obj/Cats | Cha-Sc | Anno | Download Link
---|---|---|---|---|---|---|---|---
TNO | IR-VIS | 261 | 768×576 | ❌ | few | ✔ | ❌ | [Link](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)
INO | IR-VIS | Varies | 320×240 to 512×384 | ✔ | various | ✔ | ❌ | [Link](https://www.ino.ca/en/technologies/video-analytics-dataset/videos/)
RoadScene | IR-VIS | 221 | Various | ✔ | medium | ❌ | ❌ | [Link](https://github.com/hanna-xu/RoadScene)
MSRS | IR-VIS | 1,444 | 480×640 | ✔ | abundant | ✔ | ✔ | [Link](https://github.com/Linfeng-Tang/MSRS)
LLVIP | IR-VIS | 15,488 | 1280×720 | ✔ | pedestrian / 1 | ❌ | ✔ | [Link](https://bupt-ai-cz.github.io/LLVIP/)
M3FD | IR-VIS | 4,200 | 1024×768 | ✔ | 33,603 / 6 | ✔ | ✔ | [Link](https://github.com/JinyuanLiu-CV/TarDAL)
Harvard | Med-Img | N/A | Various | ✔ | various | ❌ | ❌ | [Link](http://www.med.harvard.edu/AANLIB/home.html)
MEF | MEF | N/A | Various | ✔ | various | ❌ | ❌ | [Link](https://github.com/csjcai/SICE)
MEFB | MEF | 100 | Various | ✔ | various | ❌ | ❌ | [Link](https://github.com/xingchenzhang/MEFB)
Lytro | MF | 20 (dual-focus), 4 (triple-focus) | 520×520 | ✔ | various | ❌ | ❌ | [Link](https://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset)
MFI-WHU | MF | 120 | Various | ✔ | various | ❌ | ❌ | [Link](https://github.com/HaoZhang1018/MFI-WHU)
MFFW | MF | 19 (dual-focus) | Various | ✔ | various | ❌ | ❌ | [Link](https://www.semanticscholar.org/paper/MFFW%3A-A-new-dataset-for-multi-focus-image-fusion-Xu-Wei/4c0658f338849284ee4251a69b3c323908e62b45)
GaoFen | Pan-Sharp | N/A | Various | ✔ | various | ❌ | ❌ | [Link](https://directory.eoportal.org/web/eoportal/satellite-missions/g)
WorldView | Pan-Sharp | N/A | Various | ✔ | various | ❌ | ❌ | [Link](https://worldview.earthdata.nasa.gov/)
GeoEye | Pan-Sharp | N/A | Various | ✔ | various | ❌ | ❌ | [Link](https://earth.esa.int/eogateway/missions/geoeye-1)
QuickBird | Pan-Sharp | N/A | Various | ✔ | various | ❌ | ❌ | [Link](https://www.satimagingcorp.com/satellite-sensors/quickbird/)

## Metric


| 指標名稱   | 輸入參數 | 含意說明 |
| ---------- | -------- | -------- |
| **EN**     | image_tensor（單一圖像張量，灰階值介於0至255） | 利用圖像直方圖計算熵值，反映圖像中所包含的信息量。 |
| **CE**     | ir_img_tensor, vi_img_tensor, f_img_tensor（分別為紅外、可見光與融合圖像張量） | 通過計算二元交叉熵，衡量融合圖像與來源圖像間的相似性與信息重建效果。 |
| **QNCIE**  | ir_img_tensor, vi_img_tensor, f_img_tensor | 基於正規化交叉相關與特徵矩陣本徵值的分析，評估來源圖像與融合圖像之間的統計相關性。 |
| **TE**     | ir_img_tensor, vi_img_tensor, f_img_tensor, q（預設1）, ksize（預設256） | 利用熵值計算來源圖像與融合圖像的信息差異，反映融合過程中的信息變化。 |
| **EI**     | f_img_tensor | 使用Sobel邊緣檢測算子計算融合圖像的邊緣強度，反映邊緣信息。 |
| **SF**     | image_tensor | 通過計算圖像在水平與垂直方向上的頻率變化，評估圖像細節的豐富程度（空間頻率）。 |
| **SD**     | image_tensor | 計算圖像灰度值的標準差，反映圖像對比度與灰度分佈的離散程度。 |
| **PSNR**   | A, B, F（來源圖像A、B與融合圖像F） | 利用峰值信噪比評估融合圖像相對於來源圖像的重建品質與噪聲水平。 |
| **MSE**    | A, B, F | 計算融合圖像與來源圖像間的均方誤差，衡量像素級別的差異。 |
| **VIF**    | A, B, F | 基於多尺度模型計算視覺信息保真度，評估融合圖像中保留的視覺信息量。 |
| **CC**     | A, B, F | 通過計算相關係數，評估融合圖像與來源圖像之間的線性相關性。 |
| **SCD**    | A, B, F | 透過比較融合圖像與來源圖像間局部結構的相關性，評估融合效果的空間一致性。 |
| **Qabf**   | A, B, F | 利用外部模組（get_Qabf）獲得的融合質量指標，反映融合圖像的品質。 |
| **Nabf**   | A, B, F | 利用外部模組（get_Nabf）獲得的另一融合質量指標，用於評估融合效果。 |
| **MI**     | A, B, F, gray_level（預設256） | 互信息指標，衡量來源圖像與融合圖像間共享的信息量。 |
| **NMI**    | A, B, F, gray_level（預設256） | 歸一化互信息，將互信息標準化後評估圖像間的相似性。 |
| **AG**     | image_tensor | 計算圖像的平均梯度，反映圖像中邊緣與細節信息的強度。 |
| **SSIM**   | A, B, F | 結構相似性指數，評估融合圖像與來源圖像在結構上的相似程度。 |
| **MS-SSIM**| A, B, F | 多尺度結構相似性指數，從多個尺度評估圖像結構相似性。 |
| **Qy**     | ir_img_tensor, vi_img_tensor, f_img_tensor | 基於SSIM及局部統計特性計算的融合質量指標，重點在於局部細節的保留情況。 |
| **Qcb**    | ir_img_tensor, vi_img_tensor, f_img_tensor | 結合頻域濾波與對比度分析，評估融合圖像在頻域與對比度方面的表現。 |

## Method

### Loss

- L1-loss

### IR-VIS Fusion

- TNO
- RoadScene
- MSRS
- M3FD

### Med-image

- Harvard

### MF

- MFI-WHU

### Metric

- Entropy (EN)
- Standard Deviation (SD)
- Mutual Information (MI)
- Visual Information Fidelity (VIF)
- QABF
- Structural Similarity (SSIM)​