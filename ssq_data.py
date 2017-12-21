
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import OrderedDict
import math
from pyexcel_xls import get_data
from pyexcel_xls import save_data
import numpy as np
import cv2
from PIL import Image
import os


def get_exl_data(exl_path='C:/Users/lenovo/Desktop/ssq.xls',random_order=False):
    ssq_data=[]
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if random_order:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([
                xls_data['data'][i][9]-1,
                 xls_data['data'][i][10]-1,
                 xls_data['data'][i][11]-1,
                 xls_data['data'][i][12]-1,
                 xls_data['data'][i][13]-1,
                 xls_data['data'][i][14]-1,
                 xls_data['data'][i][8] - 1])
            ssq_data.append(temp)
    else:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([
                xls_data['data'][i][2] - 1,
                xls_data['data'][i][3] - 1,
                xls_data['data'][i][4] - 1,
                xls_data['data'][i][5] - 1,
                xls_data['data'][i][6] - 1,
                xls_data['data'][i][7] - 1,
                xls_data['data'][i][8] - 1])
            ssq_data.append(temp)

        # if xls_data['data'][i][17]>0:
        #     temp=[
        #         [xls_data['data'][i][2],
        #         xls_data['data'][i][3],
        #         xls_data['data'][i][4],
        #         xls_data['data'][i][5],
        #         xls_data['data'][i][6],
        #         xls_data['data'][i][7],
        #         34]
        #     ]
        # else:
        #     temp=[
        #         [xls_data['data'][i][2],
        #         xls_data['data'][i][3],
        #         xls_data['data'][i][4],
        #         xls_data['data'][i][5],
        #         xls_data['data'][i][6],
        #         xls_data['data'][i][7],
        #         35]
        #     ]

        # ssq_data.append(xls_data['data'][i][2])
        # ssq_data.append(xls_data['data'][i][3])
        # ssq_data.append(xls_data['data'][i][4])
        # ssq_data.append(xls_data['data'][i][5])
        # ssq_data.append(xls_data['data'][i][6])
        # ssq_data.append(xls_data['data'][i][7])#6 hong
        #
        # ssq_data.append(xls_data['data'][i][8])#+1 lan
        # if xls_data['data'][i][17]>0:
        #     ssq_data.append(100)  # 1dengjiang
        # else:
        #     ssq_data.append(101)  # 1dengjiang
        # ssq_data.append(xls_data['data'][i][19])  # 2dengjiang
        # ssq_data.insert(num,xls_data['data'][i][2])
        # ssq_data.insert(num+1, xls_data['data'][i][3])
    return ssq_data