# 数据预处理

import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

labels_dataframe = pd.read_csv("train.csv")
leave_labels = sorted(list(set(labels_dataframe["label"])))
n_classes = len(leave_labels)

class_to_num = dict(zip(leave_labels, range(n_classes)))
num_to_class = {k: v for v, k in class_to_num.items()}

class MyLeaveDataset(Dataset):
    def __init__(self, file_path, csv_path, mode="train", valid_ratio=0.2, trans=None) -> None:
        """
        Args:
            mode: train, valid, test, k_fold (not implemented)
        """
        super().__init__()
        self.file_path = file_path
        self.data_info = pd.read_csv(os.path.join(file_path, csv_path))
        self.mode = mode
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))
        self.trans = trans
        
        if mode == "train":
            self.img_arr = np.asarray(self.data_info.iloc[1: self.train_len + 1, 0])
            self.label_arr = np.asarray(self.data_info.iloc[1: self.train_len + 1, 1])
        elif mode == "valid":
            self.img_arr = np.asarray(self.data_info.iloc[self.train_len + 1: , 0])
            self.label_arr = np.asarray(self.data_info.iloc[self.train_len + 1: , 1])
        elif mode == "test":
            self.img_arr = np.asarray(self.data_info.iloc[1: , 0])
        else:
            pass
        
        self.len = len(self.img_arr)

        print(f"Init dataset mode: {mode}, valid_ratio: {valid_ratio}.")

    def __getitem__(self, index):
        image_name = self.img_arr[index]

        img = Image.open(os.path.join(self.file_path, image_name))
        img = self.trans(img)

        if self.mode == "test":
            return img
        else:
            label = self.label_arr[index]
            label = class_to_num[label]
            return img, label
        
    def __len__(self):
        return self.len
