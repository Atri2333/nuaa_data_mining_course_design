# 数据预处理

import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from sklearn import preprocessing
import cv2

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

        # img = Image.open(os.path.join(self.file_path, image_name))
        img = cv2.imread(os.path.join(self.file_path, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.trans(image=img)["image"]

        if self.mode == "test":
            return img
        else:
            label = self.label_arr[index]
            label = class_to_num[label]
            return img, label
        
    def __len__(self):
        return self.len

class MyLeaveMixUpDataset(Dataset):
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

        image_name2 = self.img_arr[self.len - index - 1]
        img2 = Image.open(os.path.join(self.file_path, image_name2))
        img2 = self.trans(img2)

        if self.mode == "test":
            return img
        else:
            label = self.label_arr[index]
            label = class_to_num[label]
            label2 = self.label_arr[self.len - index - 1]
            label2 = class_to_num[label2]
            return img, label, img2, label2
        
    def __len__(self):
        return self.len
    
class MyTrainValidDataset(Dataset):
    def __init__(self, file_path, csv_path) -> None:
        """
        Args:
            mode: train, valid, test, k_fold (not implemented yet, blame to my fucking poor wallet for autodl)
        """
        super().__init__()
        self.file_path = file_path
        self.data_info = pd.read_csv(os.path.join(file_path, csv_path))
        self.data_len = len(self.data_info.index) - 1
        

        self.img_arr = np.asarray(self.data_info.iloc[1:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[1:, 1])
        
        self.len = len(self.img_arr)

        print(f"Init TrainValidDataset")

    def __getitem__(self, index):
        image_name = self.img_arr[index]

        img = Image.open(os.path.join(self.file_path, image_name))
        img = self.trans(img)

        label = self.label_arr[index]
        label = class_to_num[label]
        return img, label
        
    def __len__(self):
        return self.len