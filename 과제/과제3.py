from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist["data"]
y = mnist["target"]

some_digit = X[0]
some_digit_img = some_digit.reshape(28, 28)

plt.imshow(some_digit_img, cmap='binary')
plt.show()

# clustering this data

k = 10
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)  # y_pred is the cluster label of each instance
print(y_pred)
print(kmeans.cluster_centers_)


score = silhouette_score(X, kmeans.labels_)
print(score)
