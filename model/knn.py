from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

from sklearn.datasets import load_iris
import model.feature_extraction as feature_extraction
from model.utils import *

def knn4hog():
    X, y = feature_extraction.HOGDataSet(".", "train.csv")
    # print(X[:10], y[:10])
    k_range = range(1, 20)
    k_scores = []

    KF = KFold(n_splits=5)

    for k in k_range:
        k_score = []
        for train_index, test_index in KF.split(X):
            # print(train_index)
            X_train, y_train = np.array(X)[train_index], np.array(y)[train_index]
            X_test, y_test = np.array(X)[test_index], np.array(y)[test_index]
            pca = PCA(n_components=100)
            pca.fit(X_train)
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)

            knn = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="kd_tree")
            knn.fit(X_train_pca, y_train)
            y_pred = knn.predict(X_test_pca)
            score = accuracy_score(y_pred, y_test)
            k_score.append(score)


        k_scores.append(sum(k_score)/5)
        print(f"k={k}, k_score:{sum(k_score)/5:.03f}")

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('acc')
    plt.show()

def knn4sift(using_local_data=True):
    X, y = feature_extraction.SIFTDataSet(".", "train.csv")
    # print(X[:2], y[:10])
    k_range = range(1, 20)
    k_scores, k_max = [], []

    KF = KFold(n_splits=5)
    X_train, X_test = [], []
    y_train, y_test = [], []
    index = 0
    for train_index, test_index in KF.split(X):
        # print(train_index)
        y_train_sub = np.array(y)[train_index]
        y_test_sub = np.array(y)[test_index]
        if using_local_data == False:
            features = np.float32([]).reshape(0, 128)
            for i in tqdm(train_index, desc=f"building wordbag{index}"):
                if X[i] is not None:
                    features = np.append(features, np.array(X[i]), axis=0)
                if i % 1000 == 0:
                    print(f"construcing wordbag: {i}")
            centers = learnVocabulary(features)
            print(f"wordbag finished")
            X_train_sub, X_test_sub = [], []
            for i in train_index:
                X_train_sub.append(calcFeatVec(X[i], centers))
            for i in test_index:
                X_test_sub.append(calcFeatVec(X[i], centers))
            
            # np.save(f"X_train_{index}.npy", np.array(X_train_sub))
            # np.save(f"X_test_{index}.npy", np.array(X_test_sub))
        else:
            X_train_sub = np.load(f"X_train_{index}.npy")
            X_test_sub = np.load(f"X_test_{index}.npy")
        X_train.append(X_train_sub)
        X_test.append(X_test_sub)
        y_train.append(y_train_sub)
        y_test.append(y_test_sub)
        index += 1


    for k in tqdm(k_range):
        k_score = []
        for i in range(5):
            knn = KNeighborsClassifier(n_neighbors=k, weights="uniform", algorithm="kd_tree")
            knn.fit(X_train[i], y_train[i])
            y_pred = knn.predict(X_test[i])
            score = accuracy_score(y_pred, y_test[i])
            k_score.append(score)

        k_scores.append(sum(k_score) / 5)
        print(f"k={k}, k_score:{sum(k_score) / 5:.03f}")

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('acc')
    plt.show()



if __name__ == "__main__":
    knn4hog()