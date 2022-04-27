import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class Decision_stump:
    def __init__(self):
        self.polarity = 1    #required to flip the predictions if error > 0.5 (explained later in detail)
        self.feature_idx = None    #we will parse through all features to find the one with the smallest error
        self.threshold = None  # we will also parse through all x in X to find suitable threshold
        self.alpha = None      #performance of our current classifier


    def predict(self, X):
        n_samples, n_features = X.shape

        X_column = X[:,self.feature_idx]     #all samples but only one col

        predictions = np.ones(n_samples)     #initialized for each sample in the beginning as 1
        if self.polarity == 1:       #default case -> all predictions where value < treshold = -1
            predictions[X_column < self.threshold] = -1
        else:                        #if polarity == -1 : flip the predictions (this will be the case if error > 0.5 )
            predictions[X_column > self.threshold] = -1
        return predictions


class Adaboost:

    def __init__(self, n_clf=5):
        # of classifiers we want to utilize, default 5
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #init weights
        w = np.full(n_samples, (1/n_samples))   #initialization with 1/n_samples in the beginning all samples have equal weight

        self.clfs = []
        for _ in range(self.n_clf):      #5 stumps in our default case
            clf = Decision_stump()

            min_error = float("inf")    #just to initialize an error that will be substituted after 1rst run with smaller error already
            for feature_i in range(n_features):
                #going through all features (greedy search)
                X_column = X[:,feature_i]
                thresholds = np.unique(X_column)      #only the unique values in the column, no need to multiple check identical thresholds
                for threshold in thresholds:
                    #greedy search over ALL (unique) values in X -> this can be computationally expensive with large datasets
                    p = 1
                    predictions = np.ones(n_samples)     #initializing all default as class +1
                    predictions[X_column < threshold] = -1    #those below the threshold value will be -1
                    missclassifies = w[y != predictions]      #then we check how many of our classifications are correct based on the logic of the 2 lines above.
                    error = sum(missclassifies)   #weights are summed up (min = 0 max = 1)
                    if error > 0.5:
                        #we flip the prediction in that case because we anticorrelate in that case
                        error = 1 - error
                        p = -1  #flip polarity and therefore the predicted -1 will be +1 and vice versa

                    if error < min_error:
                        #this is the currently best fit for our decision stump so we store polarity and threshold and feature col
                        #this will be the settings we utilize in the final stump model after lowest error is found
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i

            EPS = 1e-10       #utilized in order to not devide by 0 -> log 0 issue
            clf.alpha = 0.5 * np.log((1 - error)/(error + EPS))     #can be viewed as amount of sayingpower ? the larger the better

            predictions = clf.predict(X)

            #update weights
            w *= np.exp(-clf.alpha*y*predictions)        #update according to the saying power of the stump
            w/= np.sum(w)          #normalization in order that weights sum up again to 1 (otherwise it would not be 1 ( original weights were 1/N summing up to 1))

            self.clfs.append(clf)  #appending the properly finetuned decision trump to the list of classifiers

    def predict(self, X):
        """For each classifier we run the prediction -> alpha * predict
           Those stumps with greater error and lower alpha therefore will have less weight on the final judgment"""
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)  #will result in values with either a negative or positive sign. We are only interested in the sign of those.
        y_pred = np.sign(y_pred)
        return y_pred



def accuracy(y_pred, y_true):
    return np.sum((y_pred == y_true)/len(y_true))

data = datasets.load_breast_cancer()
X = data.data
y = data.target

y[y == 0] = -1      #dataset is 0 , 1 so we switch it to -1 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

clf = Adaboost(n_clf=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy(y_pred, y_test)
print(acc)



