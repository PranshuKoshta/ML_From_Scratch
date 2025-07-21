import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class linearRegression:
    def __init__(self, lr=0.001,epoch=1000):
        self.lr = lr
        self.epoch = epoch
    
    def fit(self, X,y):
        n_samples, n_feature = X.shape

        self.bias = 0
        self.weight = np.zeros(n_feature)


        for _ in range(self.epoch):
            y_pred = np.dot(X,self.weight) + self.bias

            dw = (1/n_samples)*np.dot(X.T,(y_pred - y))
            db = (1/n_samples)*np.sum(y_pred - y)

            self.weight -= self.lr*dw
            self.bias  -= self.lr*db

    def predict(self, X):
        y_predicted = np.dot(X,self.weight) + self.bias
        return y_predicted




if __name__ == "__main__" :
   
    x,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=123)
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=123)

    lin_reg = linearRegression(lr=0.01,epoch=1000)
    lin_reg.fit(x_train,y_train)

    predicted = lin_reg.predict(x_test)

    mse = np.mean((predicted - y_test)**2)

    print(mse)

    y_pred_line = lin_reg.predict(x)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
    plt.plot(x, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()
