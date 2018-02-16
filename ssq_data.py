
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


def get_exl_data(exl_path='./ssq.xls',random_order=False):
    ssq_data=[]
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if random_order:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i]==[]:
                break
            temp = np.asarray([
                xls_data['data'][i][9]-1,
                 xls_data['data'][i][10]-1,
                 xls_data['data'][i][11]-1,
                 xls_data['data'][i][12]-1,
                 xls_data['data'][i][13]-1,
                 xls_data['data'][i][14]-1,
                 xls_data['data'][i][8] +32])
            ssq_data.append(temp)
    else:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i]==[]:
                break
            temp = np.asarray([
                xls_data['data'][i][2] - 1,
                xls_data['data'][i][3] - 1,
                xls_data['data'][i][4] - 1,
                xls_data['data'][i][5] - 1,
                xls_data['data'][i][6] - 1,
                xls_data['data'][i][7] - 1,
                xls_data['data'][i][8] +32])
            ssq_data.append(temp)
    return ssq_data