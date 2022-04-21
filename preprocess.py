import numpy as np
import cv2
import json
import os

name_dir = os.listdir('Train_Annotations')
name_dir = [name[:-5] for name in name_dir]


for name in name_dir[:10]:
    image = cv2.imread(f'Train_Images/{name}.jpg')

    polys = []
    with open(f'Train_Annotations/{name}.json') as f:
        datas = json.load(f)
        for data in datas['shapes']:
            data = np.array(data['points'],dtype=np.int32)
            polys.append(data)
            print(np.mean(data, axis=0, dtype=np.int32))

    polys = np.array(polys)

    im = np.zeros(image.shape[:2],dtype='uint8')

    for poly in polys:
        poly = np.reshape(poly,(1,poly.shape[0],poly.shape[1]))
        cv2.polylines(im, poly, 1, 255)
        cv2.fillPoly(im,poly, 255)

    mask = im
    # cv2.namedWindow("Mask",0)
    # cv2.resizeWindow("Mask", 858, 471)
    # cv2.imshow('Mask', mask)
    masked = cv2.bitwise_and(image, image ,mask=mask)
    cv2.imwrite(f'Train_Mask/{name}.jpg', mask)
    cv2.imwrite(f'Train_Masked_Images/{name}.jpg', masked)
    # cv2.namedWindow("Masked",0)
    # cv2.resizeWindow("Masked", 858, 471)
    # cv2.imshow('Masked', masked)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
