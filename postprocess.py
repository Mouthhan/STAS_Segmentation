import cv2
import os
import numpy as np

data_path = './Unet_resnet50_bce/'
ori_path = './Public_Image/'
output_path = './result/Unet_resnet50/'
mask_path = './masked/Unet_resnet50/'
os.makedirs(output_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)

filenames = os.listdir(data_path)
for filename in filenames:
    ori_img_path = os.path.join(ori_path, filename[:-4]+'.jpg')
    img_path = os.path.join(data_path,filename)
    image = cv2.imread(img_path, 0)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ## Fill hole
    for idx in range(len(contours)):
        contour = contours[idx]
        area = cv2.contourArea(contour)
        cv2.fillPoly(image, pts =[contour], color=(255,255,255))
    
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    ## Noise Filter
    for idx in range(len(contours)):
        contour = contours[idx]
        area = cv2.contourArea(contour)
        if area > 1500:
            cv2.fillPoly(image, pts =[contour], color=(255,255,255))
        else:
            cv2.fillPoly(image, pts =[contour], color=(0,0,0))

    kernel = np.ones((3,3), np.uint8)

    ## Erase Edge
    erosion = cv2.erode(image, kernel, iterations = 2)

    # dilation = cv2.dilate(erosion, kernel, iterations = 10)
    # opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel=kernel)
    

    ori_img = cv2.imread(ori_img_path)


    # cv2.imshow('Input', image)
    # cv2.imshow('Result', dilation)
    # cv2.imshow('Erosion', erosion)
    # cv2.waitKey(0)
    # print(ori_img.shape)
    # print(image.shape)
    cv2.imwrite(os.path.join(output_path,filename), erosion)
    masked = cv2.bitwise_and(ori_img, ori_img, mask = erosion)
    cv2.imwrite(os.path.join(mask_path, filename), masked)
    