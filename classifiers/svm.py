# from sklearn.model_selection import train_test_split

# from sklearn.metrics import accuracy_score, confusion_matrix
# import numpy as np

# # def one_hot_encode(y):
# #     return np.array([[1, 0] if i == 0 else [0, 1] for i in y])

# def load_data():
#     dataset = np.loadtxt('dataset.csv', delimiter=',', dtype=np.int64)
#     X = dataset[:, :-1]
#     # y = one_hot_encode(dataset[:, -1:])
#     return train_test_split(X, y, test_size=0.25, random_state=0)

# def train(X_train, y_train):
#     svm_obj = svm.SVM()
#     svm_obj.fit(X_train, y_train)
#     return svm_obj

# def eval(obj, X_test, y_test):
#     y_pred = obj.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     print(f'Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')

# if __name__ == '__main__':
#     X_train, X_test, y_train, y_test = load_data()
#     svm_obj = train(X_train, y_train)
#     eval(svm_obj, X_test, y_test)

import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    dataset = np.loadtxt('data/result/result_0.csv', delimiter=',', dtype=str)[1:]
    dataset[:, 1] = np.where(dataset[:, 1] == 'TCP', 0,
        np.where(dataset[:, 1] == 'UDP', 1, 2))
    dataset = dataset.astype(float)
    X = dataset[:, :-1]
    X = StandardScaler().fit_transform(X)
    y = dataset[:, -1]
    return train_test_split(X, y, test_size=0.25, random_state=0)

# train_data = pd.read_csv('./classifiers/data/result/result_train_0.csv')
# test_data = pd.read_csv('./classifiers/data/result/result_test_0.csv')
X_train, X_test, y_train, y_test = load_data()
# X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
# X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
# print(len(y_train))
# X_train, y_train

svm_obj = svm.SVC()
svm_obj.fit(X_train, y_train)
y_train_pred = svm_obj.predict(X_train)
y_test_pred = svm_obj.predict(X_test)
# print(y_train_pred)
print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_test, y_test_pred))
tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
print(f'Acc: {accuracy_score(y_train, y_train_pred)} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')
print(f'F1_score: {f1_score(y_train, y_train_pred)} / Precision: {precision_score(y_train, y_train_pred)}')
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
print(f'Acc: {accuracy_score(y_test, y_test_pred)} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')
print(f'F1_score: {f1_score(y_test, y_test_pred)} / Precision: {precision_score(y_test, y_test_pred)}')
# print(X_train)