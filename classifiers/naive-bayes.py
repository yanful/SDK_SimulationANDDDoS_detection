from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def one_hot_encode(y):
    return np.array([[1, 0] if i == 0 else [0, 1] for i in y])

def load_data():
    dataset = np.loadtxt('data/result/result_0.csv', delimiter=',', dtype=str)[1:]
    dataset[:, 1] = np.where(dataset[:, 1] == 'TCP', 0,
        np.where(dataset[:, 1] == 'UDP', 1, 2))
    dataset = dataset.astype(float)
    X = dataset[:, :-1]
    X = StandardScaler().fit_transform(X)
    y = dataset[:, -1]
    return train_test_split(X, y, test_size=0.25, random_state=0)

def train(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    eval(gnb, X_train, y_train)
    return gnb

def eval(gnb, X_test, y_test):
    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')
    print(f'F1_score: {f1_score(y_test, y_pred)} / Precision: {precision_score(y_test, y_pred)}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    gnb = train(X_train, y_train)
    eval(gnb, X_test, y_test)
