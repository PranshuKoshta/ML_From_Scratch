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
   
    x,y = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=123)
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=123)

    lin_reg = linearRegression(lr=0.1,epoch=1000)
    lin_reg.fit(x_train,y_train)

    predicted = lin_reg.predict(x_test)

    mse = np.mean((predicted - y_test)**2)

    print(mse)

    # y_pred_line = lin_reg.predict(x)
    # cmap = plt.get_cmap("viridis")
    # fig = plt.figure(figsize=(8, 6))
    # m1 = plt.scatter(x_train, y_train, color=cmap(0.9), s=10)
    # m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
    # plt.plot(x, y_pred_line, color="black", linewidth=2, label="Prediction")
    # plt.show()

    # 3D plot

    fig = plt.figure(figsize=(10, 8))
    # Create a 3D subplot. The 'projection' argument is key.
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot the actual data points as a 3D scatter plot
    #    x-axis -> first feature (X_test[:, 0])
    #    y-axis -> second feature (X_test[:, 1])
    #    z-axis -> actual target value (y_test)
    ax.scatter(x_test[:, 0], x_test[:, 1], y_test, color='blue', marker='o', label='Actual Test Data')

    # 2. To plot the predicted plane, we need to create a grid of x1 and x2 values
    x1_surf = np.linspace(x_test[:, 0].min(), x_test[:, 0].max(), 10)
    x2_surf = np.linspace(x_test[:, 1].min(), x_test[:, 1].max(), 10)
    x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)
    
    # 3. Use the trained model to predict the 'y' value (z-axis) for each point on our grid
    #    We need to combine our x1_surf and x2_surf into a format the model expects
    X_surf = np.vstack((x1_surf.ravel(), x2_surf.ravel())).T
    y_surf_pred = lin_reg.predict(X_surf)
    
    # Reshape the predicted y values back to the grid shape for plotting
    y_surf_pred = y_surf_pred.reshape(x1_surf.shape)

    # 4. Plot the regression plane
    ax.plot_surface(x1_surf, x2_surf, y_surf_pred, alpha=0.5, color='red', label='Predicted Regression Plane')

    # Set labels for the 3 axes
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target Value (y)')
    ax.set_title('Multiple Linear Regression (2 Features)')
    
    # A legend is tricky for 3D plots, but we can add one conceptually
    # Note: plot_surface doesn't have a `label` argument for the legend in the same way.
    # We can create a proxy artist for the legend if needed.
    # For now, the colors are distinct enough.

    plt.show()
