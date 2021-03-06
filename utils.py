from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
import cv2
import json
import random
import wandb
from PIL import Image
import os

## Color Map

cls_color = {
    0:  [0, 0, 0],
    1:  [255, 255, 255]
}


def label2masks(pred, label, ori, idx, epoch, size, loss_type):
    '''
    Transfer label to mask
    '''
    os.makedirs(f'./Template_Valid_Result/{size}_{loss_type}', exist_ok= True)
    ## Original Image
    ori *= 255
    ori = ori.astype(np.uint8)
    ori = np.transpose(ori, (1,2,0))

    ## Pred Mask
    mask = np.empty((pred.shape[0], pred.shape[1], 3),dtype =np.uint8)
    mask[pred == 0] = cls_color[0]  # (Black: 000) Background
    mask[pred == 1] = cls_color[1]  # (White: 111) Foreground
    ## GroundTruth Mask
    mask_gt = np.empty((label.shape[0], label.shape[1], 3),dtype =np.uint8)
    mask_gt[label == 0] = cls_color[0]  # (Black: 000) Background
    mask_gt[label == 1] = cls_color[1]  # (White: 111) Foreground
    result = Image.fromarray(mask)
    result_ori = Image.fromarray(ori)
    result_gt = Image.fromarray(mask_gt)
    ## Save
    result.save(f'./Template_Valid_Result/{size}_{loss_type}/{idx}_{epoch}_pred.png')
    result_ori.save(f'./Template_Valid_Result/{size}_{loss_type}/{idx}_{epoch}_ori.png')
    result_gt.save(f'./Template_Valid_Result/{size}_{loss_type}/{idx}_{epoch}_gt.png')
    return mask

class STASDataset(Dataset):
    def __init__(self, json_root, img_root, names, size):
        self.size = size
        self.json_root = json_root
        self.img_root = img_root
        self.names = names
    
    def mask2label(self, mask):
        masks = np.empty(mask.shape[:2],dtype=np.int64)
        mask = (mask >= 128).astype(int)
        
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[mask == 7] = 1  # (White: 111) Goal: STAS
        masks[mask == 0] = 0  # (Black: 000) Other
        return masks

    def crop(self, img, x, y, img_type):
        x_fix = 0 if x + self.size < img.shape[1] else  x + self.size - img.shape[1]
        y_fix = 0 if y + self.size < img.shape[0] else  y + self.size - img.shape[0]
        x = x - x_fix
        y = y - y_fix
        if img_type == 'img':
            crop_img= img[y:y+self.size,x:x+self.size,:]
        elif img_type == 'mask':
            crop_img= img[y:y+self.size,x:x+self.size]
        return crop_img

    def __getitem__(self, idx):
        image = cv2.imread(f'{self.img_root}{self.names[idx]}.jpg')
        polys = []
        centers = []
        ## Read Json get annotations & centers
        with open(f'{self.json_root}{self.names[idx]}.json') as f:
            datas = json.load(f)
            for data in datas['shapes']:
                data = np.array(data['points'],dtype=np.int32)
                center = np.mean(data, axis=0, dtype=np.int32)
                polys.append(data)
                centers.append(center)

        polys = np.array(polys)
        centers = np.array(centers)

        im = np.zeros(image.shape,dtype='uint8')

        for poly in polys:
            poly = np.reshape(poly,(1,poly.shape[0],poly.shape[1]))
            cv2.polylines(im, poly, 1, (255,255,255))
            cv2.fillPoly(im,poly, (255,255,255))
        label = self.mask2label(im)

        choose = centers[np.random.randint(len(centers))]

        high_x = choose[0] if choose[0] - int(self.size*0.5) < 0 else int(self.size*0.5)
        high_y = choose[1] if choose[1] - int(self.size*0.5) < 0 else int(self.size*0.5)
        x = np.random.randint(high_x) 
        y = np.random.randint(high_y)

        ## Random crop or Target crop
        random_flag = random.randint(0, 9)
        if random_flag % 3 != 0:
            img = self.crop(image, choose[0] - x, choose[1] - y, 'img')
            label_crop = self.crop(label, choose[0] - x, choose[1] - y, 'mask')
        else:
            rand_x = random.randint(0,image.shape[1] - self.size -1)
            rand_y = random.randint(0,image.shape[0] - self.size -1)
            img = self.crop(image, rand_x, rand_y, 'img')
            label_crop = self.crop(label, rand_x, rand_y, 'mask')
        # img = self.crop(image, choose[0] - x, choose[1] - y, 'img')
        # label_crop = self.crop(label, choose[0] - x, choose[1] - y, 'mask')
        tfms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.ColorJitter(0.5,0.5,0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        return tfms(img), label_crop

    def __len__(self):
        return len(self.names)


## Ref: https://github.com/clcarwin/focal_loss_pytorch
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.reshape(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.reshape(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

## https://blog.csdn.net/CaiDaoqing/article/details/90457197
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

def iou_score(pred, labels):
    '''
    Compute IoU score on target class
    '''
    mean_iou = 0 

    tp_fp = np.sum(pred == 1)
    tp_fn = np.sum(labels == 1)
    tp = np.sum((pred == 1) * (labels == 1))

    iou = tp / (tp_fp + tp_fn - tp)
    mean_iou += iou
    print('IOU: %f' % mean_iou)
    return mean_iou

def dice_loss(pred, labels):
    '''
    Compute DICE loss on target class
    '''
    mean_iou = 0 

    tp_fp = np.sum(pred == 1)
    tp_fn = np.sum(labels == 1)
    tp = np.sum((pred == 1) * (labels == 1))

    iou = tp / (tp_fp + tp_fn - tp)
    mean_iou += iou
    dice = (mean_iou * 2) / (mean_iou + 1)
    return 1 - dice

def dice_score(pred, labels):
    '''
    Compute DICE score on target class
    '''
    mean_iou = 0 

    tp_fp = np.sum(pred == 1)
    tp_fn = np.sum(labels == 1)
    tp = np.sum((pred == 1) * (labels == 1))

    iou = tp / (tp_fp + tp_fn - tp)
    mean_iou += iou
    dice = (mean_iou * 2) / (mean_iou + 1)
    print('DICE: %f' % dice)
    return dice

def fix_randomseed(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    # Cuda
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

## Wandb Log
def log(train_iou, train_loss, train_dice, valid_iou, valid_loss, valid_dice, epoch):
    wandb.log({"Train IOU": train_iou,
               "Train DICE": train_dice,
               "Train Loss": train_loss,
               "Valid IOU": valid_iou,
               "Valid DICE": valid_dice,
               "Valid Loss": valid_loss,
               "Epoch": epoch})


## Wandb Log
def log_u2net(train_iou, train_target_loss, train_ensemble_loss, train_dice, valid_iou, valid_target_loss, valid_ensemble_loss, valid_dice, epoch):
    wandb.log({"Train IOU": train_iou,
               "Train DICE": train_dice,
               "Train Target Loss": train_target_loss,
               "Train Ensemble Loss": train_ensemble_loss,
               "Valid IOU": valid_iou,
               "Valid DICE": valid_dice,
               "Valid Target Loss": valid_target_loss,
               "Valid Ensemble Loss": valid_ensemble_loss,
               "Epoch": epoch})

