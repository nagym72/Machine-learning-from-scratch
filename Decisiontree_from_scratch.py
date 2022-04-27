import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from collections import Counter


def entropy(y):
    """we count the number of occurences of each feature in the respective category -> e.g number of days with rain vs days with sunshine.
    Then we compute the frequencies (p(Event) / number total events and return the entropy :
    E = - sum p*log2(p)... in this case p sum p rain equals 2 terms. """
    hist = np.bincount(y)
    ps = hist/len(y)
    return - np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    """We initialize feature, threshold, left, right, value."""
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


    def is_leaf_node(self):
        """If we are at the end of a split (E.g a leaf node) we return self.value if it is not None."""
        return self.value is not None


class Decision_Tree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        """min_samples_split: how many samples need to be min in a bin in order to go further down the road (i.e grow tree further)
           max_depth: number of iterations we compute to find all possible thresholds for best cutoff to make decisions alongside the tree.
           n_feats: if one wants to limit the number of features to < #columns(attributes)
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None


    def fit(self, X, y):
        #grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) #X.shape[1] = columns of dataset = the feature vector columns if self.n_feats was passed as argument to function call
        #we set self.n_feats to this value
        self.root = self._grow_tree(X, y) # helper function that grows the tree, (X = feature vector, y = labels), check _grow_tree for infos


    def predict(self, X):
        #traverse tree
        return np.array([self._traverse_tree(x, self.root) for x in X])



    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


    def _grow_tree(self, X, y, depth=0):
        """Takes feature vector X as input, y- label vector and depth = 0 as default."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))     #unique will return a list with only the unique class labels in the label vector y

        #stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            """if either the depth is reached -> exhausted the max depth iteration passed into func call, or we are left with only 1 label (there are only samples
            of class 1 or 0 in the node left, or there are simply not enough samples to further split (specified in func call at the beginning), then we stop and 
            look into the leaf node, count the most common instance of samples and return that label."""
            leaf_value = self._most_common_label(y)     #helper func _most_common_label see below
            return Node(value=leaf_value)               #returns a Node class object with value = the predicted label. -> we are done and finished.

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)    # random.choice takes (a, size, replace) as arguments
        # a equals the array from where random will draw elements, size= the length of this array that will be generated and replace=False means we
        #are not allowed to chose the same element twice i.e no replacement of already drawn features.

        #greedy search

        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)      #helper func that gets the best criteria to split, see _best_criteria below

        #grow children that result from split

        """We found the split that maximizes entropy gain and now execute the found split.
        """
        left_idxs , right_idxs = self._split(X[:, best_feat], best_thresh)    #split along the columnvector of given found best feature, with the best threshold to separate into left and right child nodes.
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1) # we continue down the road with the new values now being the ones that below the threshold  for all columns, the labels for those values and increase depth +1
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1) #same for right split as above for left

        return Node(best_feat, best_thresh, left, right)   #we return a Node object with best feature to split, best threshold in the feature vector column to split and the resulting left and right children nodes.


    def _most_common_label(self, y):
        """We initialize a counter with all labels (y - vector) that were passed to func call and utialize the most.common function of Counter class (imported from Collections, default)
        most_common(1)[0][0] means we return only the most frequent label (1), and this gives us a list of tuples (x,y) where x is the label and y is the number of occurences. -> [0][0] means first tuple
        and first element of tuple"""
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0] #return a list with (1) most common instances as a tuple (y, n-times) where [0] specifies the tuple and [0] the first elemenet in tuple
        return most_common   #returning the most common label. this is the decision then.

    def _best_criteria(self, X, y, feat_idxs):
        """Takes as arguments feature vector X, labels y and feat_idxs that should be taken for splitting.
        best_gain is set to -1, if a split is favoured along specific parameters then there will be an entropy gain > -1.
        We initialize split_idx and split_threshold as None and go through the feat_idxs array that we passed to this helper func.
        X_column = X[:, feat_idx] means we get all rows for the specified column (feat_idx). we collect the values of this column and get rid of duplicated (np.unique)
        We then continue to parse through all these values and calculate the information gain for each value. These information gains are then compared against each other and
        if the gain is better than -1 we substitute the better split as new gain, new split_idx and new split_threshold. We still continue to parse through thresholds and look at
        all of those possibilities, and in the end we return the best split treshold (according to information gain) and the split_idx (which column aka which feature we split)
        """

        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)      #_information_gain is a helper function to calc the information gain for each possible split scenario along each column. see below _information_gain func

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, split_thresh):
        """We pass as arguments the labels y, the feature vector column! X and the split threshold we want to check for its information gain if we would split at given threshold.
        the parent entropy is calculated via entropy func (global def function).

        In order to generate the split, we utilize a helper func _split which will do the split, afterwards we calculate the entropies for the childs (left split right split) and computethe information gain
        """
        #parent entropy
        parent_entropy = entropy(y)

        #generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)  #we obtain 2 arrays, left split node, right split node.

        if len(left_idxs) == 0 or len(right_idxs) == 0:     #if one of those is 0, there was no split!
            return 0
        #weightes avg of child entropies
        """Entropy of child = Entropy of parent - weighted child entropy
        Here we calculate the entropies for the children and then normalize them by the lengths of both arrays (left split , right split)
        """
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) *e_l + (n_r/n) *e_r

        #return information gain
        ig = parent_entropy - child_entropy # this result will be compared against the gain (default-1,) needs to be larger than -1 in order to justify given split.
        return ig


    def _split(self, X_column, split_tresh):
        """Takes as argument the feature vector column X and the value where we split.
        We split in two parts, either > threshold or <= threshold. then we flatten this into a 1D array and return both left_ifx, right_idxs for entropy calculations of children in
        _information_gain helper func."""

        left_idxs = np.argwhere(X_column <= split_tresh).flatten()
        right_idxs = np.argwhere(X_column > split_tresh).flatten()
        return left_idxs, right_idxs




def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


clf = Decision_Tree(max_depth=10)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

acc = accuracy(y_test, y_pred)

print(acc)


