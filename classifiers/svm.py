from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data():
    X = None
    y = None
    return train_test_split(X, y, test_size=0.25, random_state=0)

def train(X_train, y_train):
    svm_obj = svm.SVM()
    svm_obj.fit(X_train, y_train)
    return svm_obj

def eval(obj, X_test, y_test):
    y_pred = obj.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    svm_obj = train(X_train, y_train)
    eval(svm_obj, X_test, y_test)