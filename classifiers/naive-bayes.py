from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


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

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    gnb = train(X_train, y_train)
    eval(gnb, X_test, y_test)