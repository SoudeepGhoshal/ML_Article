# Unsupervised Learning

---

## Introduction

Unsupervised learning is a type of machine learning where the model is trained on data without labeled responses. Instead of relying on pre-defined labels, the objective is to uncover hidden patterns or intrinsic structures within the input data. This approach is particularly useful when dealing with large datasets where labeling each data point is impractical or impossible.

## Types of Unsupervised Learning:

Broadly 3 techniques of Unsupervised Learning are used:

### Clustering:

Clustering is the task of dividing a dataset into groups, or clusters, of similar items. Items in the same cluster are more similar to each other than to those in other clusters. This technique is widely used for customer segmentation, market research, and image compression. Clustering helps to find natural groupings within data, providing insights into the underlying patterns and structures.

Let's delve deeper into clustering by using the K-Means clustering algorithm on the Iris dataset. K-Means is one of the simplest and most popular unsupervised learning algorithms. It aims to partition 'n' observations into 'k' clusters in which each observation belongs to the cluster with the nearest mean.

1. Importing Libraries: We import the necessary libraries, including NumPy, Pandas, Scikit-Learn, and Matplotlib.
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
```
2. Loading the Dataset: We use the 'load_iris' function from Scikit-Learn to load the Iris dataset. The Iris dataset is used for example purpose.
```python
iris = load_iris()
X = iris.data
```
3. Applying K-Means Clustering: We initialize the K-Means clustering algorithm with 'n_clusters=3' and fit it to the dataset.
```python
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
```
4. Getting Cluster Labels: After fitting the model, we obtain the labels for each data point, indicating the cluster to which each point belongs.
```python
labels = kmeans.labels_
```
5. Reducing Dimensions for Visualization: Since visualizing high-dimensional data can be challenging, we use PCA to reduce the dataset to 2 dimensions.
```python
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```
6. Plotting the Clusters: We use Matplotlib to create a scatter plot of the data points in the reduced dimensional space, coloring them according to their cluster labels. The cluster centroids are marked with red 'X' markers. This is for visualization purpose.
```python
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```
Output:

### Dimensionality Reduction:

Dimensionality reduction is the process of reducing the number of input variables or features in a dataset while retaining as much information as possible. This technique is crucial when dealing with high-dimensional data, as it helps to mitigate the curse of dimensionality, reduce computational cost, and improve model performance. Common techniques include Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).

Let's explore dimensionality reduction using Principal Component Analysis (PCA) on the Iris dataset. PCA is a technique that transforms the data into a new coordinate system where the greatest variances by any projection of the data come to lie on the first coordinates (called principal components), the second greatest variances on the second coordinates, and so on.

1. Importing Libraries: We import the necessary libraries, including NumPy, Pandas, Scikit-Learn, and Matplotlib.
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
```
2. Loading the Dataset: We use the 'load_iris' function from Scikit-Learn to load the Iris dataset. The Iris dataset is used for example purpose.
```python
iris = load_iris()
X = iris.data
y = iris.target
```
3. Applying PCA: We initialize PCA with 'n_components=2', which means we want to reduce the data to two dimensions. We then fit PCA to the dataset and transform the data.
```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
```
4. Plotting the PCA Results: We use Matplotlib to create a scatter plot of the data points in the reduced two-dimensional space, coloring them according to their class labels.
```python
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Classes')
plt.show()
```

Output:

### Anomaly Detection:

Clustering is the task of dividing a dataset into groups, or clusters, of similar items. Items in the same cluster are more similar to each other than to those in other clusters. This technique is widely used for customer segmentation, market research, and image compression. Clustering helps to find natural groupings within data, providing insights into the underlying patterns and structures.

Let's delve deeper into clustering by using the K-Means clustering algorithm on the Iris dataset. K-Means is one of the simplest and most popular unsupervised learning algorithms. It aims to partition 'n' observations into 'k' clusters in which each observation belongs to the cluster with the nearest mean.

1. Importing Libraries: We import the necessary libraries, including NumPy, Pandas, Scikit-Learn, and Matplotlib.
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
```
2. Loading the Dataset: We use the load_iris function from Scikit-Learn to load the Iris dataset. The Iris dataset is used for example purpose.
```python
iris = load_iris()
X = iris.data
```
3. Applying K-Means Clustering: We initialize the K-Means clustering algorithm with 'n_clusters=3' and fit it to the dataset.
```python
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
```
4. Getting Cluster Labels: After fitting the model, we obtain the labels for each data point, indicating the cluster to which each point belongs.
```python
labels = kmeans.labels_
```
5. Reducing Dimensions for Visualization: Since visualizing high-dimensional data can be challenging, we use PCA to reduce the dataset to 2 dimensions.
```python
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```
6. Plotting the Clusters: We use Matplotlib to create a scatter plot of the data points in the reduced dimensional space, coloring them according to their cluster labels. The cluster centroids are marked with red 'X' markers. This is for visualization purpose.
```python
plt.figure(figsize=(10, 7))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```
Output:
