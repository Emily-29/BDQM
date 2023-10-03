import os
import cv2
import random
import math
import numpy as np
from tqdm import tqdm
import scipy.io as scio
from pandas import DataFrame
import torch


def _get_mean_std(images_path):
    img_filenames = os.listdir(images_path)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(os.path.join(images_path, img_filename))
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    mean = m[0][::-1].tolist()
    std = s[0][::-1].tolist()
    return mean, std


def _get_img_name(path):
    name = [f for f in os.listdir(path)]
    name.sort()
    return name


def _array2list(x):
    temp = []
    for i in range(len(x)):
        temp.append(x[i][0])
    return temp


def _Max_MinNormalization(data):
    _max = np.max(data)
    _min = np.min(data)
    data = (data - _min) / (_max - _min)
    return data.tolist()


def get_performance(mos, pred):
    data = DataFrame({'x': mos, 'y': pred})
    srocc = data.corr(method='spearman')
    krocc = data.corr(method='kendall')
    plcc = data.corr(method='pearson')
    t = 0
    for i in range(len(mos)):
        t += (mos[i] - pred[i]) ** 2
    return srocc.iloc[0, 1], krocc.iloc[0, 1], plcc.iloc[0, 1]


def load_dataset(data_mat):
    test_img_name = _array2list(scio.loadmat(data_mat)['test_img_name'].squeeze().tolist())
    test_img_mos = _Max_MinNormalization(scio.loadmat(data_mat)['test_img_mos'].squeeze().tolist())
    return [test_img_name, test_img_mos]


def _patches(img, patch_size):  # img->tensor: [1, c, h, w]
    _h_num = img.shape[2] // patch_size
    _w_num = img.shape[3] // patch_size
    patches = img[:, :, 0:patch_size, 0:patch_size]
    for i in range(_h_num):
        for j in range(_w_num):
            temp = img[:, :, i * patch_size:i * patch_size + patch_size, j * patch_size:j * patch_size + patch_size]
            patches = torch.cat([patches, temp], dim=0)
    return patches[1:, :, :, :]


def divide_patches(img, patch_size, state):  # img->tensor: [1, c, h, w]
    _, _, h, w = img.shape
    _h = h // patch_size * patch_size
    _w = w // patch_size * patch_size
    if state == 'train':
        choice_list = [img[:, :, h - _h:, w - _w:], img[:, :, h - _h:, :_w], img[:, :, :_h, w - _w:], img[:, :, :_h, :_w]]
        resize_img = random.choice(choice_list)
    else:
        resize_img = img[:, :, :_h, :_w]
    return _patches(resize_img, patch_size)


def np2Tensor(img, color_type='rgb', range='0_1'):
    # RGB = [255, 255, 255]
    # HSV = [180, 255, 255]
    # LAB = [255, 255, 255]
    img_np = np.array(img)
    if color_type == 'hsv':
        img_np = (cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)).astype(np.float32)
    elif color_type == 'lab':
        img_np = (cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)).astype(np.float32)
    else:  # color_type='rgb'
        img_np = img_np.astype(np.float32)

    if color_type == 'hsv':
        if range == '-1_1':
            img_np[:, :, 0] = (img_np[:, :, 0] - 90.) / 90.
            img_np[:, :, 1] = (img_np[:, :, 1] - 127.5) / 127.5
            img_np[:, :, 2] = (img_np[:, :, 2] - 127.5) / 127.5
        elif range == '0_1':
            img_np[:, :, 0] = img_np[:, :, 0] / 180.
            img_np[:, :, 1] = img_np[:, :, 1] / 255.
            img_np[:, :, 2] = img_np[:, :, 2] / 255.
        else:  # range='0_255'
            img_np = img_np / 1.
    else:  # color_type='rgb' or 'lab'
        if range == '-1_1':
            img_np = (img_np - 127.5) / 127.5
        elif range == '0_1':
            img_np = img_np / 255.
        else:  # range='0_255'
            img_np = img_np / 1.

    img_np_transpose = np.ascontiguousarray(img_np.transpose((2, 0, 1)))
    img_tensor = torch.from_numpy(img_np_transpose)
    return img_tensor


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]
