from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data():
    X = None
    y = None
    return train_test_split(X, y, test_size=0.25, random_state=0)

def train(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

def eval(gnb, X_test, y_test):
    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'Acc: {acc} / TN: {tn} / FP: {fp} / FN: {fn} / TP: {tp}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    gnb = train(X_train, y_train)
    eval(gnb, X_test, y_test)