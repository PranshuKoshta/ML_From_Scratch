

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self,X,y):
        pass
    
    def predict(self, X):
        pass




if __name__ == "__main__" :
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import pandas as pd

    iris = datasets.load_iris(as_frame=True)
    X,y = iris.data, iris.target

    X_train,y_train,X_test,y_test = train_test_split(X,y, test_size= 0.2, random_state=123 )
    