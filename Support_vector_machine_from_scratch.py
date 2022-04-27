import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=7294)
y = np.where(y == 0, -1, 1)  #original y is either 0 or 1. We transform the 0s into -1 -> result will be either -1 or 1

class SVM:
    def __init__(self, learning_rate = 0.001, lambda_par = 0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_par = lambda_par
        self.n_iters = n_iters
        self.bias = None
        self.weights = None


    def fit(self, X, y):
        #fitting the labels in order that we either have -1 or 1 labels (if label <=0 it will be set to -1).
        y_ = np.where(y <= 0, -1, 1)  #redundant because above already done
        n_samples, n_features = X.shape
        #initialization again at 0 for both weight vector and bias
        self.weights = np.zeros(n_features)
        self.bias = 0


        for _ in range(self.n_iters):
            #usually this is for n in range(epoch):
            for idx, x_i in enumerate(X):
                #for each value and idx of value in X
                condition = y_[idx] * (np.dot(x_i, self.weights)-self.bias) >= 1     #class * dotproduct of feature*weights -bias. 2 possible outcomes, either <1 or >1.
                if condition:   #true if bigger than 1
                    db = 0
                    dw = 2*self.lambda_par*self.weights
                else:
                    db = y_[idx]
                    dw = 2*self.lambda_par*self.weights - y_[idx]*x_i

                #update weights and bias
                self.weights -= self.lr * dw
                self.bias -= self.lr * db




    def predict(self, X):
        lin_output = np.dot(X, self.weights) - self.bias

        return np.sign(lin_output)


clf = SVM()
clf.fit(X, y)
pred = clf.predict(X)
#print(clf.weights, clf.bias)

def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] *x + b + offset)/ w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.scatter(X[:,0], X[:,1], marker="o", c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.weights, clf.bias, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.weights, clf.bias, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.weights, clf.bias, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.weights, clf.bias, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.weights, clf.bias, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.weights, clf.bias, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 3, x1_max + 3])

    plt.show()

visualize_svm()


def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true)/len(y_true)

print(accuracy(pred, y))  #roughly 96%