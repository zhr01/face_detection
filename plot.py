#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Hison on 12/17/16
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

beta = 2

f = open('plot.pkl','rb')
R = pickle.load(f)
P = pickle.load(f)
x = np.linspace(0.01, 0.99, 99)

plt.subplot(131)
plt.ylabel('查准率')
plt.xlabel('置信度')
plt.grid()
plt.plot(x, P)

plt.subplot(132)
plt.ylabel('查全率')
plt.xlabel('置信度')
plt.grid()
plt.plot(x, R)

line = []
for xx, yy in zip(R, P):
    v = ((1+beta**2) * xx * yy) / ((beta**2)*xx + yy)
    line.append(v)

plt.subplot(133)
plt.ylabel('F曲线')
plt.xlabel('置信度')
plt.grid()
plt.plot(x, line)

plt.show()