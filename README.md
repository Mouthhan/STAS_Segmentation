# STAS_Segmentation
腫瘤氣道擴散 Spread Through Air Spaces (STAS) 切割任務

## Dataset 
https://tbrain.trendmicro.com.tw/Competitions/Details/22

![image](https://user-images.githubusercontent.com/50419632/164417685-a0a4a0ad-93cb-44db-b516-36958b2bf47f.png)

於上方連結下載 Dataset 後，解壓縮至當前目錄，如圖所示


## Preprocess
可執行 preprocess.py 將 mask groundtruth 存出來顯示

## Change Size
目前嘗試過 input 為 [128, 224, 256, 448, 512]，可透過 train.py 中的 SIZE 更改

## Train - .py file from scratch & sliding windows (Abandoned)
可更動 train.py 中的 model_path 變更儲存的路徑，目前只嘗試 CrossEntropyLoss，預計更改為 FocalLoss+DICE Loss

### TODO
- [x] 更改為 FocalLoss
- [x] 將 Evaluation Method 從 IOU 改為 DICE (符合競賽規則)
- [x] 新增 evaluate.py 預測 testing set
- [x] 加入 DICE Loss 進行調和
- [x] 於 Valid 之中挑選部分 Visualize
- [x] 更改 sliding windows 方式(大到小微調細節)
- [x] 加入沒有 target 的影像 (調和比例，避免每個都有 target 讓 sliding windows 過多誤判斷)
- [x] 找不同 Backbone 如 U-2-NET

### DONE
1. FocalLoss 影響不大
2. 已可於 valid 中計算 IOU & DICE (但由於有 random crop，還需完成完整的 evaluate.py)
3. DICE Loss 微幅提升
4. Batch 調整為 128 明顯提升
5. 更改 sliding windows 有提升 但不顯著
6. 加入沒有 target 沒啥進步
7. U-2-Net沒啥提升 (可能缺乏 pretrained weight)

## Train - .ipynb Pretrained Encoder & Whole image

```python
pip install albumentations==1.1.0			## For Data Augmentation
pip install segmentation_models_pytorch		## Torch segmentation model implementation
```

### TODO

- [x] 比較 BCELoss & DiceLoss
- [x] 調整 init lr(1e-3 >> 3e-4) & decay 策略
- [ ] 訓練 1/3 後改為 DICE loss
- [ ] 比較不同 Encoder backbone (e.g. ResNet-101, ResNeXt-101, Efficient-b4)
- [x] 比較 Model 架構(缺Unet++)
- [ ] Ensemble Multi model

### Done

1. 純 BCELoss 表現遠高於 DiceLoss 
2. Decay策略  lr = lr * (1 -  (now_epoch / total_epochs))^0.9 (一半後)
3. 嘗試了 **DeepLab-v3++** & Unet & MANet (粗體最佳)

### Train Configs(Best)

random seed = 8863

optimizer = Adam

loss = BCELoss

epochs = 100

init_lr = 3e-4

batch_size = 6

augmentation 

```python
import albumentations as albu
## Image force resized to 800x800

def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.Rotate(limit=40,p=0.3,border_mode=cv2.BORDER_CONSTANT),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        
        albu.HueSaturationValue(p=0.6),
        albu.Sharpen(p=0.5),
        albu.RandomBrightnessContrast(p=0.4),

        albu.Crop(x_min=0, y_min=0, x_max=800, y_max=750, p=0.5),
        albu.PadIfNeeded(800, 800)

        
    ]
    return albu.Compose(train_transform)
```

```python
random_seed = 8863
epochs = 50
lr = 1e-4
criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
batch_size = 1
gradient_accum_iter = 32
warmup_step = total_step * 0.12
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_step, total_step)
```
