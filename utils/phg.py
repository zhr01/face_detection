#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 11/30/16
import cv2
import numpy as np

def compared_footprint(img1, img2):
    img1 = cv2.resize(img1, (8,8), interpolation=cv2.INTER_CUBIC)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).reshape(-1)
    img1_gray = (img1_gray / 4).astype(np.uint8)
    average_1 = img1_gray.sum() / 64
    diff_1 = img1_gray - average_1
    fp1 = []
    for d in diff_1:
        if d >=0: fp1.append(1)
        else: fp1.append(0)

    img2 = cv2.resize(img2, (8, 8), interpolation=cv2.INTER_CUBIC)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).reshape(-1)
    img2_gray = (img2_gray / 4).astype(np.uint8)
    average_2 = img2_gray.sum() / 64
    diff_2 = img2_gray - average_2
    fp2 = []
    for d in diff_2:
        if d >= 0:
            fp2.append(1)
        else:
            fp2.append(0)

    dissimilar = 0
    for x1, x2 in zip(fp1, fp2):
        if x1 != x2: dissimilar += 1

    return dissimilar


if __name__ == '__main__':
    img1 = cv2.imread('/home/zhr/tensorflow/opencv/notebook/335361958719662316.jpg')
    img2 = cv2.imread('/home/zhr/tensorflow/opencv/notebook/335361958719662316.jpg')

    d = compared_footprint(img1, img2)
    print(d)