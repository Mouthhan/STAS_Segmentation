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

        high_x = choose[0] if choose[0] - int(self.size*0.8) < 0 else int(self.size*0.8)
        high_y = choose[1] if choose[1] - int(self.size*0.8) < 0 else int(self.size*0.8)
        x = np.random.randint(high_x) 
        y = np.random.randint(high_y)


        img = self.crop(image, choose[0] - x, choose[1] - y, 'img')
        label_crop = self.crop(label, choose[0] - x, choose[1] - y, 'mask')
        # cv2.imshow('123',img)
        # cv2.imshow('mask.jpg',label_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        tfms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
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
    print('\nIOU: %f\n' % mean_iou)
    return mean_iou

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
    print('\nDICE: %f\n' % dice)
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

