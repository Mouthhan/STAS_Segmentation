import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils import STASDataset, FocalLoss, DiceLoss, iou_score, dice_score, log, fix_randomseed, label2masks
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from model import UNet
import wandb

def train(size, loss_type):
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="STAS_Segmentation", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"U-Net-{size}-gamma1-{loss_type}", 
        # Track hyperparameters and run metadata
        config={
        "learning_rate": 3e-4,
        "architecture": "UNET",
        "dataset": "Tbrain-2022-STAS-Segmentation",
        "epochs": 50,
        })
    SIZE = size
    SEED = 8863
    fix_randomseed(SEED)
    os.makedirs('./models', exist_ok=True)
    img_root = 'Train_Images/'
    json_root = 'Train_Annotations/'
    model_path = f'models/UNET_crop{SIZE}_gamma1_{loss_type}.ckpt'
    name_dir = os.listdir('Train_Annotations')
    names = [name[:-5] for name in name_dir]

    train_idx = [idx for idx in range(len(names)) if idx % 10 != 0]
    valid_idx = [idx for idx in range(len(names)) if idx % 10 == 0]

    all_dataset = STASDataset(json_root, img_root, names, SIZE)
    train_set = Subset(all_dataset, train_idx)
    valid_set = Subset(all_dataset, valid_idx)

    BATCH_SIZE = 2
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size = 1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(3, 2).to(device)


    # summary(model, (3, 224, 224), device="cuda")

    EPOCHS = 50
    lr = 3e-4
    decay = 5e-4
    accum_iter = 32
    early_stop = 50
    best_dice = 0.0

    if 'CE' in loss_type:
        criterion = nn.CrossEntropyLoss()
    elif 'Focal' in loss_type:
        criterion = FocalLoss(gamma=1, alpha=0.3)
    creterion_dice = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    print(f'Start Training with size: {SIZE}, Loss: {loss_type}')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = []
        train_iou = 0
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
                if 'Mix' in loss_type:
                    loss_dice = creterion_dice(logits.argmax(dim=1), labels)
                    loss = loss * (EPOCHS - epoch) * 0.02 + loss_dice * epoch * 0.02
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
        train_iou = iou_score(preds, masks)
        train_dice = dice_score(preds, masks)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{EPOCHS:03d} ] Loss = {train_loss:.5f}, IOU = {train_iou:.4f}, DICE = {train_dice:.4f}")

        # ---------- Validation ----------
        
        valid_loss = []
        valid_iou = 0
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
                if 'Mix' in loss_type:
                    loss_dice = creterion_dice(logits.argmax(dim=1), labels)
                    loss = loss * (EPOCHS - epoch) * 0.02 + loss_dice * epoch * 0.02
                pred = logits.cpu().argmax(dim=1).numpy()
                valid_preds += list(pred)
                valid_masks += list(labels.cpu().numpy())
                valid_loss.append(loss.item())
                if idx % 10 == 0 and (epoch+1) % 10 == 0:
                    result = label2masks(pred[0], labels.cpu()[0].numpy(), imgs.cpu()[0].numpy(),idx, epoch+1, SIZE, loss_type)
                # valid_mious.append(miou)
            # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_preds = np.array(valid_preds)
        valid_masks = np.array(valid_masks)
        # print(valid_preds.shape)
        # print(valid_masks.shape)
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_iou = iou_score(valid_preds, valid_masks)
        valid_dice = dice_score(valid_preds, valid_masks)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{EPOCHS:03d} ] Loss = {valid_loss:.5f}, IOU = {valid_iou:.4f}, DICE = {valid_dice:.4f}")
        log(train_iou, train_loss, train_dice, valid_iou, valid_loss, valid_dice, epoch + 1)
        
        # Save Best Model
        if valid_dice > best_dice:
            torch.save(model.state_dict(),model_path)
            print(f'Save Model With DICE: {valid_dice:.4f}')
            best_dice = valid_dice
            stop_count = 0
        else:
            stop_count += 1
            if stop_count > early_stop:
                break
    wandb.finish()
    print('End of Training')

if __name__=='__main__':
    sizes = [128,256]
    loss_types = ['Focal','Focal-Mix']
    for size in sizes:
        for loss_type in loss_types:
            train(size, loss_type)
    
