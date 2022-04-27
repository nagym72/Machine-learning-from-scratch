import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()       #120 entries with 4 attributes
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#print(X_train.shape)     #120 x 4 size
#print(X_train[0])
#
#print(y_train.shape)     #120 entries corresponding to the 3 flower types denoted with [0,1,2]
#print(y_train)

plt.figure()

plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolors="k", s=20)       #show first col and all entries at x axis X[:,0] ]and all entries and col 2 X[:,1] as y-axis
plt.show()

def eucledian_distance(x1, x2):
    """Takes two arrays as inputs and returns a single float value sqrt(diff between vectors squared and then summed up)"""
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    """K-nearest neighbour works unsupervised
        default initialization with 3 clusters k=3 (3 flower types in the dataset)
        """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]      #helper function below utilized
        return np.array(predicted_labels)

    def _predict(self, x):
        #compute distances, get k-nearest neighbours, labels
        #majority vote: get most common class label.

        #for each x (shape = 1x4) consisting of 1 row and 4 columns we compute the eucledian distance
        #example:
        # first entry in x : [4.1 2.2 3.3 4.4]  shape = 1x4
        #eucledian distance would be :
        # sqrt(Summation over (([4.1 2.2 3.3 4.4] - [5.1 2.5 3.0 1.1])**2))) -> 4 differences squared and summed up then sqrt returns 1 value.
        # This is done for each x a total of 120 times (total 120*120)
        distances = [eucledian_distance(x, x_train) for x_train in self.X_train]      # explained above. returns list with the eucledian distance values
        k_indices = np.argsort(distances)[:self.k]        #np.argsort will return the indices from smallest to largest in the distance list. from those 120 ranks we will fish out the first 3 [:selfk]
        k_nearest_labels = [self.y_train[i] for i in k_indices]    #corresponding to the 3 indexes of the entries that had the smallest eucledian distances towards the sample.
        most_common = Counter(k_nearest_labels).most_common(1)  #returns a tuple with the actual label (flowertype [0,1,2] and its frequency e.g 5  -> (2,5) for e.g flowertype 2 counted 5 times
        return most_common[0][0] #out of those tuples we are simply interested in the y label [0][0]

clf = KNN(k=7)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  np.sum(predictions == y_test) / len(y_test)            #diff between predicted labels and actual labels

print(acc)

