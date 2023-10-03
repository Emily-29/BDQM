import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.utilities import np2Tensor, crop_center


class MyDataSet(Dataset):
    def __init__(self,
                 img_path,
                 split_data,
                 use_augmentation=False,
                 resize_size=False,
                 iscrop=False,
                 ):
        super(MyDataSet, self).__init__()

        self.img_path = img_path
        self.use_augmentation = use_augmentation
        self.resize_size = resize_size
        self.iscrop = iscrop

        self.img_name = split_data[0]  # split_data -> (test_img_name, test_img_mos)
        self.img_mos = split_data[1]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_name[index]))
        if self.iscrop:
            img = crop_center(np.array(img), self.resize_size[0], self.resize_size[1])
        elif self.resize_size is not None:
            img = img.resize((self.resize_size[0], self.resize_size[1]))

        if self.use_augmentation:
            if random.random() <= 0.5:
                img = img.rotate(random.choice([90, 180, 270]), expand=True)
            if random.random() <= 0.5:
                img = img.transpose(random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]))

        img = np2Tensor(img, color_type='rgb', range='0_1')  # color_type='rgb' or 'hsv' or 'lab', range='0_1' or '-1_1' or '0_255'
        label = torch.from_numpy(np.asarray([self.img_mos[index]], dtype=np.float32))
        return img, label

    def __len__(self):
        return len(self.img_name)
