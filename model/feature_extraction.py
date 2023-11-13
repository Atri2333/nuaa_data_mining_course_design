# used for traditional ml
from skimage.io import imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from model.utils import *

csv_path = "train.csv"
labels_dataframe = pd.read_csv(csv_path)
leave_labels = sorted(list(set(labels_dataframe["label"])))
n_classes = len(leave_labels)

class_to_num = dict(zip(leave_labels, range(n_classes)))
num_to_class = {k: v for v, k in class_to_num.items()}

def getHOG(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = resize(gray, (128, 64))
    # resized_img /= 255.0
    # imshow(resized_img)
    fd = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), block_norm="L2")
    return normalization(fd)

def HOGDataSet(file_path, csv_path):
    data_info = pd.read_csv(os.path.join(file_path, csv_path))
    data_len = len(data_info.index) - 1
    print(f"data_len:{data_len}")
    X, y = [], []
    for i in tqdm(range(data_len), desc="loading dataset"):
        image_name = data_info.iloc[i+1, 0]
        img = cv2.imread(os.path.join(file_path, image_name))
        x = getHOG(img)
        label = data_info.iloc[i+1, 1]
        label = class_to_num[label]
        X.append(x)
        y.append(label)
        if i % 1000 == 0:
            pass   
    return X, y

def getSIFT(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = resize(gray, (224, 224))
    resized_img = cv2.normalize(resized_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    sift = cv2.SIFT_create()
    _, des = sift.detectAndCompute(resized_img, None)
    return des

def SIFTDataSet(file_path, csv_path):
    data_info = pd.read_csv(os.path.join(file_path, csv_path))
    data_len = len(data_info.index) - 1
    print(f"data_len:{data_len}")
    X, y = [], []
    for i in tqdm(range(data_len), desc="loading dataset"):
        image_name = data_info.iloc[i+1, 0]
        img = cv2.imread(os.path.join(file_path, image_name))
        x = getSIFT(img)
        label = data_info.iloc[i+1, 1]
        label = class_to_num[label]
        X.append(x)
        y.append(label)
        if i % 1000 == 0:
            pass   
    return X, y

# def getORB(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     resized_img = resize(gray, (224, 224))
#     resized_img = cv2.normalize(resized_img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
#     orb = cv2.ORB_create()
#     _, des = orb.detectAndCompute(resized_img, None)
#     return des

# def ORBDataSet(file_path, csv_path):
#     data_info = pd.read_csv(os.path.join(file_path, csv_path))
#     data_len = len(data_info.index) - 1
#     print(f"data_len:{data_len}")
#     X, y = [], []
#     for i in range(data_len):
#         image_name = data_info.iloc[i+1, 0]
#         img = cv2.imread(os.path.join(file_path, image_name))
#         x = getORB(img)
#         label = data_info.iloc[i+1, 1]
#         label = class_to_num[label]
#         X.append(x)
#         y.append(label)
#         if i % 1000 == 0:
#             print(i, label)    
#     return X, y