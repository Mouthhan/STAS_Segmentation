import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import json


SIZE = 448
class STASDataset(Dataset):
    def __init__(self, json_root, img_root, names):
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
        x_fix = 0 if x + SIZE < img.shape[1] else  x + SIZE - img.shape[1]
        y_fix = 0 if y + SIZE < img.shape[0] else  y + SIZE - img.shape[0]
        x = x - x_fix
        y = y - y_fix
        if img_type == 'img':
            crop_img= img[y:y+SIZE,x:x+SIZE,:]
        elif img_type == 'mask':
            crop_img= img[y:y+SIZE,x:x+SIZE]
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

        high_x = choose[0] if choose[0] - int(SIZE*0.8) < 0 else int(SIZE*0.8)
        high_y = choose[1] if choose[1] - int(SIZE*0.8) < 0 else int(SIZE*0.8)
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