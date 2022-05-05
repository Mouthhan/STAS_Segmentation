import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils import STASDataset, FocalLoss, DiceLoss, iou_score, dice_score, log, fix_randomseed, label2masks,log_u2net
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from model import UNet
import wandb
from U2Net import U2NETP

def train(size, loss_type):
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="STAS_Segmentation", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"U-2-NetP-{size}-{loss_type}-randomcrop03-norm", 
        # Track hyperparameters and run metadata
        config={
        "learning_rate": 3e-4,
        "architecture": "U-2-NETP",
        "dataset": "Tbrain-2022-STAS-Segmentation",
        "epochs": 50,
        })
    SIZE = size
    SEED = 8863
    fix_randomseed(SEED)
    os.makedirs('./models', exist_ok=True)
    img_root = 'Train_Images/'
    json_root = 'Train_Annotations/'
    model_path = f'models/U2NETP_crop{SIZE}_{loss_type}_randomcrop03_norm.ckpt'
    name_dir = os.listdir('Train_Annotations')
    names = [name[:-5] for name in name_dir]

    train_idx = [idx for idx in range(len(names)) if idx % 10 != 0]
    valid_idx = [idx for idx in range(len(names)) if idx % 10 == 0]

    all_dataset = STASDataset(json_root, img_root, names, SIZE)
    train_set = Subset(all_dataset, train_idx)
    valid_set = Subset(all_dataset, valid_idx)
    if SIZE == 128:
        BATCH_SIZE = 8
    elif SIZE == 256:
        BATCH_SIZE = 4
    else:  
        BATCH_SIZE = 2
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size = 1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = UNet(3, 2).to(device)
    model = U2NETP(3, 2).to(device)


    # summary(model, (3, 224, 224), device="cuda")

    EPOCHS = 50
    lr = 3e-4
    decay = 5e-4
    accum_iter = 32 // BATCH_SIZE
    early_stop = 50
    best_dice = 0.0

    if 'CE' in loss_type:
        criterion = nn.CrossEntropyLoss()
    elif 'Focal' in loss_type:
        criterion = FocalLoss(gamma=1, alpha=0.3)
    creterion_dice = DiceLoss()
    # optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    print(f'Start Training with size: {SIZE}, Loss: {loss_type}')
    for epoch in range(EPOCHS):
        model.train()
        train_target_loss = []
        train_ensemble_loss = []
        train_iou = 0
        optimizer.zero_grad()
        preds = []
        masks = []
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            with torch.set_grad_enabled(True):
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                # logits = model(imgs)
                logits = model(imgs)

                ## merge all loss
                for idx, logit in enumerate(logits):
                    if idx == 0:
                        total_loss = criterion(logit,labels) 
                        loss = criterion(logit,labels)
                        if 'Mix' in loss_type:
                            total_loss_dice = creterion_dice(logit.argmax(dim=1), labels)
                            loss_dice = creterion_dice(logit.argmax(dim=1), labels)
                    else:
                        total_loss += criterion(logit,labels)
                        if 'Mix' in loss_type:
                            total_loss_dice += creterion_dice(logit.argmax(dim=1), labels)
                    loss = loss * (1 - epoch * 0.01)  + loss_dice * epoch * 0.02
                    total_loss = total_loss * (1 - epoch * 0.01) + total_loss_dice * epoch * 0.02
                
                # if 'Mix' in loss_type:
                #     loss_dice = creterion_dice(logits.argmax(dim=1), labels)
                #     loss = loss * (EPOCHS - epoch) * 0.02 + loss_dice * epoch * 0.02
                train_target_loss.append(loss.detach().item())
                train_ensemble_loss.append(total_loss.detach().item())
                total_loss = total_loss / accum_iter
                total_loss.backward()
                

                preds += list(logits[0].cpu().argmax(dim=1).numpy())
                masks += list(labels.cpu().numpy())
                

            ## Update weights
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
        preds = np.array(preds)
        masks = np.array(masks)
        # print(preds.shape)
        # print(masks.shape)
        train_target_loss = sum(train_target_loss) / len(train_target_loss)
        train_ensemble_loss = sum(train_ensemble_loss) / len(train_ensemble_loss)
        train_iou = iou_score(preds, masks)
        train_dice = dice_score(preds, masks)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{EPOCHS:03d} ] Target Loss = {train_target_loss:.5f}, Ensemble Loss = {train_ensemble_loss:.5f}, IOU = {train_iou:.4f}, DICE = {train_dice:.4f}")

        # ---------- Validation ----------
        
        valid_target_loss = []
        valid_ensemble_loss = []
        valid_iou = 0
        valid_preds = []
        valid_masks = []
        # Iterate the validation set by batches.
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_loader)):
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)

                ## merge all loss
                for idx, logit in enumerate(logits):
                    if idx == 0:
                        total_loss = criterion(logit,labels) 
                        loss = criterion(logit,labels)
                        if 'Mix' in loss_type:
                            total_loss_dice = creterion_dice(logit.argmax(dim=1), labels)
                            loss_dice = creterion_dice(logit.argmax(dim=1), labels)
                    else:
                        total_loss += criterion(logit,labels)
                        if 'Mix' in loss_type:
                            total_loss_dice += creterion_dice(logit.argmax(dim=1), labels)
                    loss = loss * (1 - epoch * 0.01)  + loss_dice * epoch * 0.02
                    total_loss = total_loss * (1 - epoch * 0.01) + total_loss_dice * epoch * 0.02
                
                pred = logits[0].cpu().argmax(dim=1).numpy()
                valid_preds += list(pred)
                valid_masks += list(labels.cpu().numpy())
                valid_target_loss.append(loss.item())
                valid_ensemble_loss.append(total_loss.item())
                if batch_idx % 10 == 0 and (epoch+1) % 10 == 0:
                    result = label2masks(pred[0], labels.cpu()[0].numpy(), imgs.cpu()[0].numpy(),batch_idx, epoch+1, SIZE, loss_type)
                # valid_mious.append(miou)
            # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_preds = np.array(valid_preds)
        valid_masks = np.array(valid_masks)
        # print(valid_preds.shape)
        # print(valid_masks.shape)
        valid_target_loss = sum(valid_target_loss) / len(valid_target_loss)
        valid_ensemble_loss = sum(valid_ensemble_loss) / len(valid_ensemble_loss)
        valid_iou = iou_score(valid_preds, valid_masks)
        valid_dice = dice_score(valid_preds, valid_masks)
        scheduler.step(valid_dice)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{EPOCHS:03d} ] Target Loss = {valid_target_loss:.5f}, Ensemble Loss = {valid_ensemble_loss:.5f}, IOU = {valid_iou:.4f}, DICE = {valid_dice:.4f}")
        log_u2net(train_iou, train_target_loss, train_ensemble_loss, train_dice, valid_iou, valid_target_loss, valid_ensemble_loss, valid_dice, epoch + 1)
        
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
    sizes = [256]
    loss_types = ['CE-Mix']
    for size in sizes:
        for loss_type in loss_types:
            train(size, loss_type)
    
