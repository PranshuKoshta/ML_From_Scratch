import numpy as np
from collections import Counter

def euclean_distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [ self.predictor(x) for x in X ]
        return np.array(y_pred)

    def predictor(self,x):
        # compute distance
        distances = [ euclean_distance(x,x_train) for x_train in self.X_train ]

        # find k nearest neighbors
        k_nearest = np.argsort(distances)[:self.k]

        k_nearest_labels = [ self.y_train[i] for i in k_nearest]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common





if __name__ == "__main__" :
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import pandas as pd

    digits = datasets.load_digits()
    X,y = digits.data, digits.target

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state=123)

    clf = KNN()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)

    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(accuracy)


