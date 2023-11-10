from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import numpy as np

csv_path = "train.csv"
labels_dataframe = pd.read_csv(csv_path)
leave_labels = sorted(list(set(labels_dataframe["label"])))
n_classes = len(leave_labels)

class_to_num = dict(zip(leave_labels, range(n_classes)))
num_to_class = {k: v for v, k in class_to_num.items()}


def getDataSet(file_path, csv_path):
    data_info = pd.read_csv(os.path.join(file_path, csv_path))
    data_len = len(data_info.index) - 1
    print(f"data_len:{data_len}")
    X, y = [114514] * data_len, []
    for i in range(data_len):
        label = data_info.iloc[i+1, 1]
        label = class_to_num[label]
        y.append(label)
    return X, y

def Classify(X, y):
    KF = KFold(n_splits=5)
    score_max = 0
    for train_index, test_index in KF.split(y):
        # print(train_index)
        X_train, y_train = np.array(X)[train_index], np.array(y)[train_index]
        X_test, y_test = np.array(X)[test_index], np.array(y)[test_index]
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_pred = dummy.predict(X_test)
        score = accuracy_score(y_pred, y_test)
        if score_max < score:
            score_max = score
    print(f"DummyClassifier acc:{score_max:.03f}")

if __name__ == "__main__":
    X, y =  getDataSet(".", csv_path)
    Classify(X, y)