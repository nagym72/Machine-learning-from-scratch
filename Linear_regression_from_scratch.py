import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)  #split in test and training set at 20% cutoff (80% training)


#fig = plt.figure(figsize=(8,6))
#plt.scatter(X[:,0],y, color ="b", marker="o", s=30)

#plt.show()

class LinearRegeression:
    """Alpha = learning rate
        n_iters = number of iterations
        weights and bias are initialized at None (but could also be initialized at 0 e.g)"""

    def __init__(self, alpha=0.01, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) #here we set them all to 0. This is a vector while bias is simply a float
        self.bias = 0 #bias is 0 as well in the beginning.

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias       # summation over all [x*w for all x and w]  + bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))   #X.transposed has shape 1x80, y has shape 80  dot is simply multiplication followed by summation dot is matrix product
            db = (1/n_samples) * np.sum(y_predicted - y)

            #update step (would be torch.optim.step() if optimizer = Stochastic gradient descent in our case)
            self.weights -= self.alpha*dw
            self.bias -= self.alpha*db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

regressor = LinearRegeression(alpha=0.01)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

def MSE(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted)**2)

mse_value = MSE(y_test, predicted)
print(mse_value)      #should be as small as possible ( ideal case (no diff between actual and predicted y ) = 0 )

y_predline = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_predline, color="black", linewidth=2, label="Prediction")
plt.show()
