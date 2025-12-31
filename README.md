### AI CUP 2025秋季賽－電腦斷層心臟肌肉影像分割競賽 I－心臟肌肉影像分割 - 環境配置與模組說明

本文件說明如何重現本專案之執行環境，並詳列各階段核心模組的輸入與輸出資料流，以利第三方使用者進行除錯 (Debug) 與驗證。

## 🛠 一、安裝與環境配置 (Installation & Configuration)

本專案建議於 **Linux (Ubuntu 20.04+)** 環境下執行，並需要支援 CUDA 的 NVIDIA GPU。

### 1. 建立虛擬環境
建立並啟用獨立的 Conda 環境以避免套件衝突：
```bash
conda create -n aicup_heart python=3.9
conda activate aicup_heart
```
### 2. 安裝相依套件
請依照您的 CUDA 版本選擇對應的 PyTorch (建議 2.0+ 版本)：
```bash
# 例如 CUDA 11.8
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 安裝 nnU-Net 核心與其他依賴
pip install nnunetv2
pip install -r requirements.txt
```
### 3. 設定環境變數
nnU-Net 嚴格依賴環境變數來定位資料。請在終端機執行以下指令，或寫入 ~/.bashrc：
```bash
export nnUNet_raw="/您的專案路徑/nnUNet_raw"
export nnUNet_preprocessed="/您的專案路徑/nnUNet_preprocessed"
export nnUNet_results="/您的專案路徑/nnUNet_results"
```
### 4. 必備資料結構
```text
nnUNet_raw/
├── Dataset501_Heart/
    ├── dataset.json          # 定義標籤對應 (Background: 0, Muscle: 1)
    ├── imagesTr/             # 訓練集影像 (格式：.nii.gz, 檔名需含 _0000)
    ├── labelsTr/             # 訓練集標註 (格式：.nii.gz)
    └── imagesTs/             # 測試集影像 (格式：.nii.gz)
```
## ⚙️ 重要模塊輸出/輸入 (Important Modules I/O)

下表詳列本專案各階段核心程式的資料流向，若執行過程中發生錯誤，請優先檢查該步驟的「輸入路徑」檔案是否存在且格式正確。

| 模塊名稱 | 執行指令 / 腳本 | 輸入 (Input) | 輸出 (Output) | 功能與除錯備註 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 格式轉換** | `convert_dicom.py`<br>*(自訂腳本)* | **原始 DICOM 資料夾**<br>(競賽提供的原始格式) | **NIfTI 影像 (.nii.gz)**<br>存放於 `imagesTr/` | 將 DICOM 序列轉換為單一 3D NIfTI 檔，並統一座標方向。若失敗請檢查 DICOM Header 是否完整。 |
| **2. 資料規劃與前處理** | `nnUNetv2_plan_and_preprocess` | **原始 NIfTI 影像**<br>`nnUNet_raw/Dataset501_Heart` | **前處理數據 (.npz)**<br>**規劃檔 (plans.json)**<br>存放於 `nnUNet_preprocessed/` | 分析影像 Spacing 與強度分佈，並執行裁切 (Cropping) 與重取樣。若訓練速度過慢，請檢查輸出的 `.npz` 是否過大。 |
| **3. 模型訓練** | `nnUNetv2_train` | **前處理數據 (.npz)**<br>`nnUNet_preprocessed/` | **模型權重 (.pth)**<br>**訓練日誌 (training_log.txt)**<br>存放於 `nnUNet_results/` | 執行 U-Net 訓練迴圈。輸出包含 `checkpoint_best.pth` (最佳權重) 與 `checkpoint_final.pth`。 |
| **4. 模型推論** | `nnUNetv2_predict` | **測試集影像 (.nii.gz)**<br>`imagesTs/`<br>**模型權重 (.pth)** | **預測結果 (.nii.gz)**<br>指定之輸出資料夾 | 使用滑動視窗 (Sliding Window) 進行預測。若結果有拼接痕跡，需檢查 Gaussian Overlap 參數。 |
| **5. 後處理 (選用)** | `post_processing.py` | **預測結果 (.nii.gz)** | **最終提交檔案**<br>(符合競賽規範格式) | 執行最大連通域保留 (Keep Largest Component) 以去除細微雜訊，並轉換檔名以符合繳交規範。 |

快速驗證步驟 (Quick Start)
確認環境配置無誤後，您可以用以下順序執行整個流程：
```text
前處理：nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres
訓練：nnUNetv2_train 501 3d_fullres 0 (訓練 Fold 0)
預測：nnUNetv2_predict -i nnUNet_raw/Dataset501_Heart/imagesTs -o output -d 501 -c 3d_fullres -f 0
```


    
