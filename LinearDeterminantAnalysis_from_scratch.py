import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None


    def fit(self, X, y):
        n_features = X.shape[1] #150 , 4 again the iris dataset utilized so [1] = 4
        class_labels = np.unique(y)   #[0, 1, 2] for the 3 flower types

        # calculate the scatter matrix Sb(between class) and Sw(within)
        mean_overall = np.mean(X, axis=0)  #returns means of all 4 columns
        S_w = np.zeros((n_features, n_features)) #4x4    will be filled WITHIN the class
        S_b = np.zeros((n_features, n_features)) #4x4    will be filled BETWEEN classes

        for c in class_labels:
            X_c = X[y == c]       #only those fetched that correspond to label y -> 3 groups of flowers between 150
            mean_c = np.mean(X_c, axis=0)    #mean within the classlabel y

            # fills the S_within.  X_c - mean_c.T.dot with itself is
            # 4, n_c  * n_c, 4 -> result shape: 4, 4
            S_w += (X_c - mean_c).T.dot((X_c - mean_c))     #4x4 shape
            n_c = X_c.shape[0] #50 for each class
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1) #(4,) needs to be (4,1)
            S_b += n_c * (mean_diff).dot(mean_diff.T)

        A = np.linalg.inv(S_w).dot(S_b)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0:self.n_components]



    def transform(self, X):
        # we project data onto new components
        # 150x4 * 4xn_components ( self.linear_discriminants = n_comp x 4 so .T is required to allow .dot)
        return np.dot(X, self.linear_discriminants.T)
        #returns 150xn_comp vector


data = datasets.load_iris()
X = data.data
y = data.target

lda = LDA(2)       #2 components selected to keep
lda.fit(X,y)       # fitting gets us the 2x4 vector containing the eigenvectors
X_projected = lda.transform(X)     #transform data from 150x4 to 150x2
print("shape of X:", X.shape)
print("shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]      #first and sec col to show (in our case we  have 150x2 so makes sense)
x2 = X_projected[:, 1]

#x1 = X[:,1]         intresting for the case of n_components 3
#x2 = X[:,3]

plt.scatter(x1, x2, c=y, edgecolors="none", alpha=0.8, cmap= plt.cm.get_cmap("viridis", 3))

plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.colorbar()
plt.show()
