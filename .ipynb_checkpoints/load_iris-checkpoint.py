from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

def load_iris_dataframe():
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(iris.target.shape[0], -1) 
    
    X_and_y = np.concatenate((X, y), axis=1)
    columns = iris.feature_names[:]
    columns.append('target')
    
    df = pd.DataFrame(X_and_y, columns=columns)
    df.target = df.target.astype(np.int64)
    return df
    