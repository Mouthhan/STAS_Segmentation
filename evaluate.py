import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import label2masks
from model import UNet

SIZE = 512
STEP = SIZE // 2

class STAS_Eval_Dataset(Dataset):
    def __init__(self, img_root, names, size):
        self.size = size
        self.img_root = img_root
        self.names = names

    def __getitem__(self, idx):
        image = cv2.imread(f'{self.img_root}{self.names[idx]}.jpg')
        height = (942 // SIZE) + 1
        width = (1716 // SIZE) + 1
        pad_right = width * SIZE - 1716
        pad_down = height * SIZE - 942
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad([0, 0, pad_right, pad_down])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])
        return tfms(image), self.names[idx]

    def __len__(self):
        return len(self.names)

cls_color = {
    0:  [0, 0, 0],
    1:  [255, 255, 255]
}

def label2masks(pred, name, threshold):
    '''
    Transfer label to mask
    '''
    os.makedirs(f'./Test_Result/th{threshold}_{SIZE}_step{STEP}_random/', exist_ok= True)
    ## Pred Mask
    mask = np.empty((pred.shape[0], pred.shape[1], 3),dtype =np.uint8)
    mask[pred == 0] = cls_color[0]  # (Black: 000) Background
    mask[pred == 1] = cls_color[1]  # (White: 111) Foreground

    result = Image.fromarray(mask)

    ## Save
    result.save(f'./Test_Result/th{threshold}_{SIZE}_step{STEP}_random/{name}.png')

    return mask

def evaluate(threshold):

    os.makedirs('./models', exist_ok=True)
    img_root = 'Public_Image/'
    # json_root = 'Train_Annotations/'
    model_path = f'models/UNET_crop{SIZE}_gamma1_Focal-Mix.ckpt'
    name_dir = os.listdir('Public_Image')
    names = [name[:-4] for name in name_dir]


    all_dataset = STAS_Eval_Dataset(img_root, names, SIZE)

    BATCH_SIZE = 1
    test_loader = DataLoader(all_dataset, batch_size = BATCH_SIZE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(3, 2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)


    # summary(model, (3, 224, 224), device="cuda")
    height = (942 // STEP)
    width = (1716 // STEP)

    print('Start Testing')
    # ---------- Validation ----------
    
    valid_preds = []
    # Iterate the validation set by batches.
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            imgs, name = batch
            batch_logits = np.zeros((2,1024,1792))
            batch_pred = np.zeros((1024,1792))
            for h in range(height):
                for w in range(width):
                    temp = imgs[:,:,h * STEP:h * STEP + SIZE, w * STEP: w * STEP + SIZE]
                    temp = temp.to(device)
                    logits = model(temp)
                    logits = torch.softmax(logits,dim=1, dtype=float)
                    logits = torch.where(logits > threshold, logits, 0.)
                    batch_logits[:,h * STEP:h * STEP + SIZE, w * STEP: w * STEP + SIZE] += logits.cpu().numpy()[0]
                    # pred = logits.cpu().argmax(dim=1).numpy()
                    
                    # batch_pred[h * SIZE:(h + 1) * SIZE, w * SIZE: (w + 1) * SIZE] = pred
            batch_pred = batch_logits.argmax(axis=0)
            result = label2masks(batch_pred[:942,:1716], name[0], threshold)


            # pred = logits.cpu().argmax(dim=1).numpy()


    print('End of Testing')

if __name__=='__main__':
    THRESHOLDS = [0.6,0.7,0.8]
    for THRESHOLD in THRESHOLDS:
        evaluate(THRESHOLD)
