#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 12/16/16
import pickle
import scipy.io
import cv2
import os
import json
from scipy.misc import imread
from evaluate import add_rectangles

gt_path = "/home/zhr/tensorflow/crowd_count/dataset/mall_dataset/mall_gt.mat"
test_path = "raw_results.pkl"
dataset_path = "/home/zhr/tensorflow/crowd_count/dataset/mall_dataset/frames"
hypes_file = './hypes/overfeat_rezoom.json'

with open(hypes_file, 'r') as f:
    H = json.load(f)

data = scipy.io.loadmat(gt_path)["frame"][0]

pk_file = open(test_path, 'rb')
bbox = pickle.load(pk_file)
bconf = pickle.load(pk_file)
pk_file.close()

files = os.listdir(dataset_path)
files.sort()

fgbg = cv2.createBackgroundSubtractorMOG2()
for i in range(20):
    img = imread(os.path.join(dataset_path, files[i]))
    fgbg.apply(img)

WIDTH = 640
HEIGHT = 480
DIVIDE = 4


def judge_pos(rect):
    x1, y1, x2, y2 = rect
    if y2 < HEIGHT / DIVIDE:
        return 1
    elif y1 < HEIGHT / DIVIDE:
        return 2
    else:
        return 3


def ifinside(rect, points):
    x1, y1, x2, y2 = rect

    for point in points:
        cx, cy = point
        if ((cx - x1) * (cx - x2) <= 0) and ((cy - y1) * (cy - y2) <= 0):
            return True, point

    return False, []


def evaluate(confidence=0.1):
    total_positive = 0
    total_gt = 0
    true_positive = 0

    for i, (np_pred_boxes, np_pred_confidences) in enumerate(zip(bbox, bconf)):
        # img = imread(os.path.join(dataset_path, files[i]))
        # fgbg.apply(img)
        # bg = fgbg.getBackgroundImage()

        gt_points = data[i][0][0][0]
        total_gt += len(gt_points)
        for p in gt_points:
            if p[1] < HEIGHT / DIVIDE:
                total_gt -= 1

        _, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                  use_stitching=True, rnn_len=H['rnn_len'], min_conf=confidence,
                                  show_suppressed=False)

        for rect in rects:
            type = judge_pos(rect)
            if type == 1:
                continue
            elif type == 2:
                flag, p = ifinside(rect, gt_points)
                if flag:
                    total_positive += 1
                    true_positive += 1

            else:
                total_positive += 1
                flag, p = ifinside(rect, gt_points)
                if flag:
                    true_positive += 1

        # if i % 20 == 0:
        #     print("TP=%d, total_positive=%d, gt=%d, P=%f, R=%f" % (true_positive, total_positive, total_gt,
        #                                                            float(true_positive) / float(total_positive),
        #                                                            float(true_positive) / float(total_gt)))

    if total_positive == 0: precision = 1
    else: precision = float(true_positive) / float(total_positive)
    recall = float(true_positive) / float(total_gt)
    return precision, recall


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np
    import pickle
    PN = 100

    xx = []
    yy = []
    # xx.append(0.0)
    # xx.append(1.0)
    # yy.append(1.0)
    # yy.append(0.0)
    i = 0
    x_axis = np.linspace(1/PN, 1-(1/PN), PN-1)
    for x in x_axis:
        i += 1
        print(i)
        P, R = evaluate(x)
        xx.append(R)
        yy.append(P)

    f = open('plot.pkl', 'wb')
    pickle.dump(xx, f, -1)
    pickle.dump(yy, f, -1)
    f.close()

    xx_ = sorted(xx)
    yy_ = []
    for num in xx_:
        n = xx.index(num)
        yy_.append(yy[n])
    plt.ylabel('Precison')
    plt.xlabel('Recall')
    plt.grid()
    plt.plot(xx_, yy_)
    plt.show()

    # x_axis = np.linspace(0, 1, PN+1)
    line = []
    for x, y in zip(xx, yy):
        line.append((2*x*y)/(x+y))
    plt.ylabel('socre')
    plt.xlabel('confidence')
    plt.grid()
    ax = plt.subplot(1, 1, 1)
    ax.plot(x_axis, line)
    plt.show()

