import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from collections import Counter
import matplotlib.pyplot as plt

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        #mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        #covariance matrix
        cov = np.cov(X, rowvar=False)

        #eigenvectors and values
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T


        #sorting
        idxs = np.argsort(eigenvalues)[::-1]


        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        #store first n eigenvectors

        self.components = eigenvectors[0:self.n_components]


    def transform(self, X):
         X = X - self.mean
         return np.dot(X, self.components.T)


data = datasets.load_iris()

X = data.data
y = data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print(X.shape)
print(X_projected.shape)

x1 = X_projected[:,0]
x2 = X_projected[:,1]
#x3 = X_projected[:,2]   case with 3 components
val = np.zeros(len(x1))

plt.scatter(
    x1,x2, c=y, edgecolor="none", alpha=1, cmap=plt.cm.get_cmap("viridis", 3))

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
