from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

import model.feature_extraction as feature_extraction
from model.utils import *

def svm4hog():
    X, y = feature_extraction.HOGDataSet(".", "train.csv")
    # print(X[:10], y[:10])
    k_score = []

    KF = KFold(n_splits=5)
    for train_index, test_index in KF.split(X):
        # print(train_index)
        X_train, y_train = np.array(X)[train_index], np.array(y)[train_index]
        X_test, y_test = np.array(X)[test_index], np.array(y)[test_index]
        pca = PCA(n_components=200)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        svc = SVC(kernel='rbf', C=1)
        svc.fit(X_train_pca, y_train)
        
        # score = accuracy_score(y_pred, y_test)
        y_pred = svc.predict(X_test_pca)
        score = accuracy_score(y_pred, y_test)
        k_score.append(score)
        print(f"score: {score:.03f}")
    k_mean = sum(k_score) / 5
    k_max = max(k_score)

    print(f"acc for svm4hog is mean{k_mean:.03f}, max{k_max:.03f}")

def svm4sift():
    X, y = feature_extraction.SIFTDataSet(".", "train.csv")
    # print(X[:2], y[:10])
    k_score = []

    KF = KFold(n_splits=5)
    X_train, X_test = [], []
    y_train, y_test = [], []
    index = 0
    for train_index, test_index in KF.split(X):
        # print(train_index)
        y_train_sub = np.array(y)[train_index]
        y_test_sub = np.array(y)[test_index]
        features = np.float32([]).reshape(0, 128)
        for i in tqdm(train_index, desc=f"building wordbag{index}"):
            if X[i] is not None:
                features = np.append(features, np.array(X[i]), axis=0)
            if i % 1000 == 0:
                pass
        centers = learnVocabulary(features)
        print(f"wordbag finished")
        X_train_sub, X_test_sub = [], []
        for i in train_index:
            X_train_sub.append(calcFeatVec(X[i], centers))
        for i in test_index:
            X_test_sub.append(calcFeatVec(X[i], centers))
        X_train.append(X_train_sub)
        X_test.append(X_test_sub)
        y_train.append(y_train_sub)
        y_test.append(y_test_sub)
        index += 1


    for i in range(5):
        svc = SVC(kernel='rbf', C=1)
        svc.fit(X_train[i], y_train[i])
        # score = accuracy_score(y_pred, y_test)
        y_pred = svc.predict(X_test[i])
        score = accuracy_score(y_pred, y_test[i])
        k_score.append(score)
        print(f"score: {score:.03f}")
    k_mean = sum(k_score) / 5
    k_max = max(k_score)

    print(f"acc for svm4sift is mean{k_mean:.03f}, max{k_max:.03f}")


if __name__ == "__main__":
    svm4sift()