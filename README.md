# STAS_Segmentation
腫瘤氣道擴散 Spread Through Air Spaces (STAS) 切割任務

## Dataset 
https://tbrain.trendmicro.com.tw/Competitions/Details/22
於上方連結下載 Dataset 後，解壓縮至當前目錄，
![image](https://user-images.githubusercontent.com/50419632/164417685-a0a4a0ad-93cb-44db-b516-36958b2bf47f.png)
如上圖所示

## Preprocess
可執行 preprocess.py 將 mask groundtruth 存出來顯示

## Change Size
目前嘗試過 input 為 224x224 or 448x448，可透過 utils.py 中的 SIZE 更改

## Train
可更動 train.py 中的 model_path 變更儲存的路徑，目前只嘗試 CrossEntropyLoss，預計更改為 FocalLoss

## TODO
- [ ] 更改為 FocalLoss
- [ ] 將 Evaluation Method 從 Mean IOU 改為 DICE (符合競賽規則)
- [ ] 新增 evaluate.py 完整評分 valid set (all set 中 % 10 == 0) 
