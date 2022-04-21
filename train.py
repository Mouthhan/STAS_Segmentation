import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils import STASDataset
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from mean_iou_evaluate import mean_iou_score, read_masks
from model import UNet

img_root = 'Train_Images/'
json_root = 'Train_Annotations/'
model_path = 'models/UNET_crop448.ckpt'
name_dir = os.listdir('Train_Annotations')
names = [name[:-5] for name in name_dir]

train_idx = [idx for idx in range(len(names)) if idx % 10 != 0]
valid_idx = [idx for idx in range(len(names)) if idx % 10 == 0]

all_dataset = STASDataset(json_root, img_root, names)
train_set = Subset(all_dataset, train_idx)
valid_set = Subset(all_dataset, valid_idx)

BATCH_SIZE = 2
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size = BATCH_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(3, 2).to(device)


# summary(model, (3, 224, 224), device="cuda")

EPOCHS = 50
lr = 3e-4
decay = 5e-4
accum_iter = 32
early_stop = 50
best_miou = 0.0

weight = [0.5, 2.5]
class_weights = torch.FloatTensor(weight).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

print('Start Training')
for epoch in range(EPOCHS):
    model.train()
    train_loss = []
    train_miou = 0
    optimizer.zero_grad()
    preds = []
    masks = []
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        with torch.set_grad_enabled(True):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            
            loss = criterion(logits,labels)
            train_loss.append(loss.detach().item())
            loss = loss / accum_iter
            loss.backward()
            

            preds += list(logits.cpu().argmax(dim=1).numpy())
            masks += list(labels.cpu().numpy())
            

        ## Update weights
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
    preds = np.array(preds)
    masks = np.array(masks)
    # print(preds.shape)
    # print(masks.shape)
    train_loss = sum(train_loss) / len(train_loss)
    train_miou = mean_iou_score(preds, masks)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{EPOCHS:03d} ] Loss = {train_loss:.5f}, MIOU = {train_miou:.4f}")

    # ---------- Validation ----------
    
    valid_loss = []
    valid_miou = 0
    valid_preds = []
    valid_masks = []
    # Iterate the validation set by batches.
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valid_loader)):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)

            loss = criterion(logits, labels)
            pred = logits.cpu().argmax(dim=1).numpy()
            valid_preds += list(pred)
            valid_masks += list(labels.cpu().numpy())

            valid_loss.append(loss.item())
            # valid_mious.append(miou)
        # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_preds = np.array(valid_preds)
    valid_masks = np.array(valid_masks)
    # print(valid_preds.shape)
    # print(valid_masks.shape)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_miou = mean_iou_score(valid_preds, valid_masks)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{EPOCHS:03d} ] Loss = {valid_loss:.5f}, MIOU = {valid_miou:.4f}")
    # Save Best Model
    if valid_miou > best_miou:
        torch.save(model.state_dict(),model_path)
        print(f'Save Model With MIOU: {valid_miou:.4f}')
        best_miou = valid_miou
        stop_count = 0
    else:
        stop_count += 1
        if stop_count > early_stop:
            break

print('End of Training')
