import torch
from sklearn.decomposition import PCA
import numpy as np
import cv2

def get_device():
    ''' get gpu device '''
    return "cuda" if torch.cuda.is_available() else "cpu"

def model_freeze(model):
    ''' used for finetuning '''
    for param in model.parameters():
        param.requires_grad = False

def normalization(data):
    '''01 normalization'''
    data = np.array(data)
    # mu = np.mean(data)
    # sigma = np.std(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

def learnVocabulary(features):
    '''construct vocabulary by using kmeans'''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)

    flags = cv2.KMEANS_RANDOM_CENTERS

    _, __, centers = cv2.kmeans(features, 64, None, criteria, 20, flags)

    return centers

def calcFeatVec(features, centers):
    featVec = np.zeros(64)
    if features is None:
        return featVec
    for i in range(features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (64, 1)) - centers
        sqSum = (diffMat ** 2).sum(axis=1)
        dist = sqSum ** 0.5
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]
        featVec[idx] += 1
    return featVec

def rand_bbox(size, lam):
    '''cut random bbox'''
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# def pca_data(X, dim):
#     '''garbage function, i am shabi!!!'''
#     pca_1 = PCA(n_components=dim)
#     X = pca_1.fit_transform(X)
#     return X