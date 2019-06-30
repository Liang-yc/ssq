
import tensorflow as tf
# import matplotlib.pyplot as plt
from collections import OrderedDict
import math
from pyexcel_xls import get_data
from pyexcel_xls import save_data
import numpy as np
import cv2
from PIL import Image
import os

def get_exl_data_by_period(exl_path='./ssq.xls',random_order=False,use_resnet=False,times=10):
    ssq_data = []
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if use_resnet:
        if random_order:
            for i in range(2, len(xls_data['data'])-times):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == [] or xls_data['data'][i+times] == []:
                    break
                temp=[]
                for j in range(times):
                    temp.append([
                    [xls_data['data'][i+j][9] - 1],
                    [xls_data['data'][i+j][10] - 1],
                    [xls_data['data'][i+j][11] - 1],
                    [xls_data['data'][i+j][12] - 1],
                    [xls_data['data'][i+j][13] - 1],
                    [xls_data['data'][i+j][14] - 1],
                    [xls_data['data'][i+j][8] + 32],
                    [xls_data['data'][i+j][15] / 10000000000],
                    [xls_data['data'][i+j][16] / 10000000000],
                    [xls_data['data'][i+j][17] / 10000000],
                    [xls_data['data'][i+j][19] / 10000000],
                    [xls_data['data'][i+j][21] / 10000000],
                    [xls_data['data'][i+j][23] / 10000000],
                    [xls_data['data'][i+j][25] / 10000000],
                    [xls_data['data'][i+j][27] / 10000000]
                ])

                # print(temp.shape)
                #
                ssq_data.append(np.asarray(temp))
        else:
            for i in range(2, len(xls_data['data'])-times):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == [] or xls_data['data'][i+times] == []:
                    break
                temp=[]

                for j in range(times):

                    temp.append([
                    [xls_data['data'][i+j][2] - 1],
                    [xls_data['data'][i+j][3] - 1],
                    [xls_data['data'][i+j][3] - 1],
                    [xls_data['data'][i+j][5] - 1],
                    [xls_data['data'][i+j][6] - 1],
                    [xls_data['data'][i+j][7] - 1],
                    [xls_data['data'][i+j][8] + 32],
                    [xls_data['data'][i+j][15] / 10000000000],
                    [xls_data['data'][i+j][16] / 10000000000],
                    [xls_data['data'][i+j][17] / 10000000],
                    [xls_data['data'][i+j][19] / 10000000],
                    [xls_data['data'][i+j][21] / 10000000],
                    [xls_data['data'][i+j][23] / 10000000],
                    [xls_data['data'][i+j][25] / 10000000],
                    [xls_data['data'][i+j][27] / 10000000]
                ])
                ssq_data.append(np.asarray(temp))
        return ssq_data

    if random_order:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i] == []:
                break
            temp = np.asarray([
                xls_data['data'][i][9] - 1,
                xls_data['data'][i][10] - 1,
                xls_data['data'][i][11] - 1,
                xls_data['data'][i][12] - 1,
                xls_data['data'][i][13] - 1,
                xls_data['data'][i][14] - 1,
                xls_data['data'][i][8] + 32])
            ssq_data.append(temp)
    else:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i] == []:
                break
            temp = np.asarray([
                xls_data['data'][i][2] - 1,
                xls_data['data'][i][3] - 1,
                xls_data['data'][i][4] - 1,
                xls_data['data'][i][5] - 1,
                xls_data['data'][i][6] - 1,
                xls_data['data'][i][7] - 1,
                xls_data['data'][i][8] + 32])
            ssq_data.append(temp)
    return ssq_data

def get_red(exl_path='./ssq.xls', random_order=False):
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
                 xls_data['data'][i][14]-1])
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
                xls_data['data'][i][7] - 1])
            ssq_data.append(temp)
    return ssq_data

def get_blue(exl_path='./ssq.xls',use_resnet=False):
    ssq_data=[]
    xls_data = get_data(exl_path)
    if use_resnet:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([[[
                xls_data['data'][i][8] - 1]]])
            ssq_data.append(temp)
    else:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([
                xls_data['data'][i][8]-1])
            ssq_data.append(temp)
    return ssq_data

def get_red161(exl_path='./ssq.xls',use_resnet=True):
    ssq_data=[]
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if use_resnet:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([[
                [xls_data['data'][i][9]],
                 [xls_data['data'][i][10]],
                 [xls_data['data'][i][11]],
                 [xls_data['data'][i][12]],
                 [xls_data['data'][i][13]],
                 [xls_data['data'][i][14]]]])
            # print(temp.shape)
            #
            ssq_data.append(temp)
        return ssq_data

def get_exl_data(exl_path='./ssq.xls',random_order=False,use_resnet=False):
    ssq_data=[]
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if use_resnet:
        if random_order:
            for i in range(2, len(xls_data['data'])):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == []:
                    break
                temp = np.asarray([[
                    [xls_data['data'][i][9]-1],
                     [xls_data['data'][i][10]-1],
                     [xls_data['data'][i][11]-1],
                     [xls_data['data'][i][12]-1],
                     [xls_data['data'][i][13]-1],
                     [xls_data['data'][i][14]-1],
                     [xls_data['data'][i][8]+32]]])
                # print(temp.shape)
                #
                ssq_data.append(temp)
        else:
            for i in range(2, len(xls_data['data'])):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == []:
                    break
                temp = np.asarray([[
                    [xls_data['data'][i][2] - 1],
                    [xls_data['data'][i][3] - 1],
                    [xls_data['data'][i][4] - 1],
                    [xls_data['data'][i][5] - 1],
                    [xls_data['data'][i][6] - 1],
                    [xls_data['data'][i][7] - 1],
                    [xls_data['data'][i][8] + 32]]])
                ssq_data.append(temp)
        return ssq_data

    if random_order:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i] == []:
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
            if xls_data['data'][i] == []:
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




def get_dlt_data(exl_path='./dlt.xls',random_order=False,use_resnet=False):
    ssq_data=[]
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if use_resnet:
        if random_order:
            for i in range(2, len(xls_data['data'])):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == []:
                    break
                temp = np.asarray([[
                    [xls_data['data'][i][9]-1],
                     [xls_data['data'][i][10]-1],
                     [xls_data['data'][i][11]-1],
                     [xls_data['data'][i][12]-1],
                     [xls_data['data'][i][13]-1],
                     [xls_data['data'][i][14]-1],
                     [xls_data['data'][i][8]+32]]])
                # print(temp.shape)
                #
                ssq_data.append(temp)
        else:
            for i in range(2, len(xls_data['data'])):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == []:
                    break
                temp = np.asarray([[
                    [xls_data['data'][i][2] - 1],
                    [xls_data['data'][i][3] - 1],
                    [xls_data['data'][i][4] - 1],
                    [xls_data['data'][i][5] - 1],
                    [xls_data['data'][i][6] - 1],
                    [xls_data['data'][i][7] - 1],
                    [xls_data['data'][i][8] + 32]]])
                ssq_data.append(temp)
        return ssq_data

    if random_order:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i] == []:
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
            if xls_data['data'][i] == []:
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



def get_dlt_data(exl_path='./dlt2.xls',random_order=False,use_resnet=False):
    ssq_data=[]
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if use_resnet:
        if random_order:
            for i in range(2, len(xls_data['data'])):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == []:
                    break
                temp = np.asarray([[
                    [xls_data['data'][i][9]-1],
                     [xls_data['data'][i][10]-1],
                     [xls_data['data'][i][11]-1],
                     [xls_data['data'][i][12]-1],
                     [xls_data['data'][i][13]-1],
                     [xls_data['data'][i][14]+34],
                     [xls_data['data'][i][15]+34]]])
                # print(temp.shape)
                #
                ssq_data.append(temp)
        else:
            for i in range(2, len(xls_data['data'])):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == []:
                    break
                temp = np.asarray([[
                    [xls_data['data'][i][2] - 1],
                    [xls_data['data'][i][3] - 1],
                    [xls_data['data'][i][4] - 1],
                    [xls_data['data'][i][5] - 1],
                    [xls_data['data'][i][6] - 1],
                    [xls_data['data'][i][7] + 34 ],
                    [xls_data['data'][i][8] + 34]]])
                ssq_data.append(temp)
        return ssq_data

    if random_order:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i] == []:
                break
            temp = np.asarray([
                xls_data['data'][i][9]-1,
                 xls_data['data'][i][10]-1,
                 xls_data['data'][i][11]-1,
                 xls_data['data'][i][12]-1,
                 xls_data['data'][i][13]-1,
                xls_data['data'][i][14] + 34,
                xls_data['data'][i][15] + 34])
            ssq_data.append(temp)
    else:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i] == []:
                break
            temp = np.asarray([
                xls_data['data'][i][2] - 1,
                xls_data['data'][i][3] - 1,
                xls_data['data'][i][4] - 1,
                xls_data['data'][i][5] - 1,
                xls_data['data'][i][6] - 1,
                xls_data['data'][i][7] + 34,
                xls_data['data'][i][8] + 34])
            ssq_data.append(temp)
    return ssq_data
