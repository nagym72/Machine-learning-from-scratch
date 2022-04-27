import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets




class NaiveBayes:
    """X is a feature vector that contain rows x columns where rows tells how many samples were investigated and columns
    corresponds to the number of features per investigated sample is available. e.g 120x10 would mean we have 120 datasamples
    and each of those 120 has 10 attributes/features making a total of 1200 numbers in a 120x10 array
    """

    def fit(self, X, y):
        #X.shape gives e.g (120,10) means we have 120 samples and for each sample we have 10 features.
        n_samples, n_features = X.shape
        #np.unique retrieves the unique entries in an array as a list. In our case we have a binary classification
        #array which has only 0 and 1ns.. therefore we will retrieve a list [0,1] with len 2.
        self._classes = np.unique(y)
        #number of uniques classes is 2 in our case
        n_classes = len(self._classes)

        #init mean variants and priors
        #self._mean = (2x10) vector, initialized for each class 10 means along each column.
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        #self._var = (2x10) vector, initialized for each class 10 variances along each column.
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        #self._priors = is (2,) 1d array, contains 2 entries, corresponding to the prior prob. of each class
        #in our case this is 2.
        self._priors = np.zeros(n_classes, dtype=np.float64)

        #for each class in our class_ array we loop through and compute all means/variances for the class
        for c in self._classes:
            #class 0 is a (403,10) vector means, we have 403 samples belonging to class 0 with 10 features per sample
            #class 1 is a (397,10)vector, which sums up to 800 total (n_samples was 800)
            X_c =X[c==y]
            #self._mean[c,:] = row for class 0 -> has 10 0s from initializiation, we calculate the 10 mean values along
            #the column (axis=0) and set those 10 values thenn into the mean array. Same done for _variances
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            #probability for priors equals the frequency of observing sample(class x) / total samples
            #X_c.shape will return (403,10) which means 403 samples with 10 features each.
            #X_c[0] = 403 in this case.  self_priors[0] = 403/800, self_priors[1] = 397/800
            self._priors[c] = X_c.shape[0] / float(n_samples)


    def predict(self, X):
        #self._predict(x) for x in X will take each row in X (200,10) which means 10 elements per row (1 row 10 columns) and pass that
        #to the helper function. there we calculate for each 10 entries in the row the 10 means and 10 variances and report back 10 posteriors
        y_predict = [self._predict(x) for x in X]
        return y_predict


    def _predict(self, x):
        """We use a helper function to calculate for each x in X the posterior probabilities.
        For idx, c in enumerate(self._classes) will give back the class (either 0 or 1) and c (which will be either 0 or 1 as well?)
        """
        posteriors = []
        #print(self._classes)
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional =np.sum(np.log(self._probability_density(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
            #print(posteriors[np.argmax(posteriors)])

        return self._classes[np.argmax(posteriors)]

    def _probability_density(self,class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        #print(x.shape, mean.shape, var.shape)
        #numerator will give back 10 class conditionals given that it takes into account mean/var of each column(feature) of a given sample.
        numerator = np.exp(- (x - mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi*var)

        return numerator / denominator


def accuracy(y_true, y_predict):
    accuracy = np.sum(y_true == y_predict) / len(y_true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=7294)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7294)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Final accuracy on the testset is :", accuracy(y_test, predictions))




