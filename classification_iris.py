# -*- coding: utf-8 -*-
"""classification_iris.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lPZZ4kAvcwbbMGDrxuzDXhDMlbrmoKl1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris_dataset = load_iris()
iris_dataset.keys()

iris_dataset['DESCR']

iris_dataset.data.shape

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)

X_train, X_test, Y_train, Y_test

iris_df = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_df

from pandas.plotting import scatter_matrix

grr = scatter_matrix(iris_df, c=Y_train, figsize=(15, 15), marker='o',
 hist_kwds={'bins': 20}, s=60, alpha=.8)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, Y_train)

### a sample to test the model

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction_x_new = knn.predict(x_new)

prediction_x_new, iris_dataset['target_names'][prediction_x_new], iris_dataset['target_names']

### evaluation of the model


y_pred = knn.predict(X_test)

y_pred

test_score = np.mean(y_pred==Y_test)
test_score

