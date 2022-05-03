# STAS_Segmentation
腫瘤氣道擴散 Spread Through Air Spaces (STAS) 切割任務

## Dataset 
https://tbrain.trendmicro.com.tw/Competitions/Details/22

![image](https://user-images.githubusercontent.com/50419632/164417685-a0a4a0ad-93cb-44db-b516-36958b2bf47f.png)

於上方連結下載 Dataset 後，解壓縮至當前目錄，如圖所示


## Preprocess
可執行 preprocess.py 將 mask groundtruth 存出來顯示

## Change Size
目前嘗試過 input 為 224x224 or 448x448，可透過 train.py 中的 SIZE 更改

## Train
可更動 train.py 中的 model_path 變更儲存的路徑，目前只嘗試 CrossEntropyLoss，預計更改為 FocalLoss

## TODO
- [x] 更改為 FocalLoss:
- [x] 將 Evaluation Method 從 IOU 改為 DICE (符合競賽規則)
- [x] 新增 evaluate.py 預測 testing set
- [x] 加入 DICE Loss 進行調和
- [x] 於 Valid 之中挑選部分 Visualize
- [ ] 更改 sliding windows 方式(大到小微調細節)
- [ ] 加入沒有 target 的影像 (調和比例，避免每個都有 target 讓 sliding windows 過多誤判斷)
- [ ] 找不同 Backbone 如 U-2-NET

## DONE
1. FocalLoss 影響不大
2. 已可於 valid 中計算 IOU & DICE (但由於有 random crop，還需完成完整的 evaluate.py)
3. DICE Loss 微幅提升
4. Batch 調整為 128 明顯提升
