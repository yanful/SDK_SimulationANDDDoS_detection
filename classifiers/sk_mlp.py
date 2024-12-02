from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    eval(mlp, X_train, y_train)
    return mlp

def eval(mlp, X_test, y_test):
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    mlp = train(X_train, y_train)
    eval(mlp, X_test, y_test)