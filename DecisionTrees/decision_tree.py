import numpy as np

class DecisionTree:

    def __init__(self, X, level=0, features=[]):
        '''
        featureNum is number of features of X
        s.leaf is wheather or not it is a node
        prior
        leftTree is the Gawd dang left tree
        rightTree is the other one
        thresh is zero
        :param X:
        '''
        self.level = level
        self.feature_num = len(X[0])
        self.leaf = False
        self.prior = -1
        self.leftTree = None
        self.rightTree = None
        self.threshold = 0
        self.i = -1
        self.f = features
        if features == []:
            self.f = []
            for i in range(self.feature_num):
                self.f.append(i)

    @staticmethod
    def entropy(y):
        """
        assuming multiple classes, how would you implement?
        H(S ) = -pC*log2 pC,sum all potential classes
        """
        if len(y) == 0:
            return 1
        classes = {}
        for i in y:
            if i not in classes:
                classes[i] = 1
            else:
                classes[i] += 1
        if len(classes) == 1:
            return 0
        total_potential_classes = len(y)
        from math import log
        x = [classes[i] / total_potential_classes * log(classes[i] / total_potential_classes, len(classes)) for i in
             classes]
        return -sum(x)

    def information_gain(self, X, y, idx, thresh):
        """
        takes in the dataset and labels on the current leaf,
        then splits them on the threshold and index. once split,
        it returns the information gain based on current entropy
        minus the average entropy of each leaf.
        """
        _, y_left, _, y_right = self.split(X, y, idx, thresh)
        return self.entropy(y) - (len(y_left) * self.entropy(y_left) + len(y_right) * self.entropy(y_right)) / (len(y))

    def split(self, X, y, idx, thresh):
        """
        Looks at each datapoints indx, if it is less than
        the threshold, than that data is going to the left
        tree with its labels, else it is going right.
        """
        Xl, yl, Xr, yr = [], [], [], []
        for i in range(len(y)):
            if X[i][idx] < thresh:
                Xl.append(X[i])
                yl.append(y[i])
            else:
                Xr.append(X[i])
                yr.append(y[i])
        return Xl, yl, Xr, yr

    def segmenter(self, X, y):
        """
        return the feature and the threshold for the split that
        has maximum gain
        """
        thresholds = self.index_thresh(X, y)
        topThreshes = [self.information_gain(X, y, i, thresholds[i]) for i in range(self.feature_num)]
        indx = topThreshes.index(max(topThreshes))
        return topThreshes.index(max(topThreshes)), thresholds[indx]

    def index_thresh(self, X, y):
        '''
        :returns an array of the mean of each feature,
        used for a linear decision boundry to form a split.
        '''
        XL = np.array([X[i] for i in range(len(X)) if y[i] == 0])
        XR = np.array([X[i] for i in range(len(X)) if y[i] == 1])
        return (np.mean(XL, axis=0) + np.mean(XR, axis=0)) / 2.0

    def fit(self, X, y, depth):
        """
        In the case this is a leaf:
        if there are less than 7 objects, or .3 entropy (90%+ one class)
        or finally we have hit max depth, than we simply call it a leaf.
        the we get its thresholds and index
        """
        if len(y) < 7 or self.entropy(y) < 0.30 or depth <= self.level:
            self.leaf = True
            mean_y = np.mean(y)
            if mean_y <= 0.5:
                self.prior = 0
            else:
                self.prior = 1
            return 0

        self.index, self.threshold = self.segmenter(X, y)

        Xl, yl, Xr, yr = self.split(X, y, self.index, self.threshold)
        # No gain was made
        if len(yl) == 0 or len(yr) == 0:
            self.leaf = True
            mean_y = np.mean(y)
            if mean_y <= 0.5:
                self.prior = 0
            else:
                self.prior = 1
            return 0
        self.leftTree = DecisionTree(Xl, self.level + 1, features=self.f)
        self.rightTree = DecisionTree(Xr, self.level + 1, features=self.f)
        x = self.leftTree.fit(Xl, yl, depth)
        x = self.rightTree.fit(Xr, yr, depth)
        return x

    def predict(self, x):
        """
        If not a leaf
        Go left if Xi < self.threshold
        else return prior
        """
        if self.leaf:
            return self.prior
        if x[self.index] <= self.threshold:
            return self.leftTree.predict(x)
        return self.rightTree.predict(x)

    def __repr__(self, level=0):
        """
        a recursive structure that prints a good visual of
        the current tree. reccomend when printing a tree
        to go no further than depth 5, otherwise hard to see which
        features are more relevant than others.
        """
        lines, _, _, _ = self._display_aux()
        s = ''
        for line in lines:
            s += line + '\n'
        return s

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        if self.leaf:
            line = '%s' % "Class:" + str(self.prior)
            width = len(line)
            height = 1
            middle = width // 2

            return [line], width, height, middle
        left, n, p, x = self.leftTree._display_aux()
        right, m, q, y = self.rightTree._display_aux()
        s = '%s' % "{}:".format(self.f[self.index]) + str('%.2f' % (self.threshold))
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

##############################################################################################
#
# Random Forest uses Decision Trees, so both go in this file
# 
##############################################################################################
class RandomForest():
    
    def __init__(self, X, numForests, features=[]):
        """
        creates numForests worth of DecisionTrees
        """
        self.forests = []
        for i in range(numForests):
            self.forests.append(DecisionTree(X, features=features))
        self.num = numForests


    def fit(self, X, y, depth, train_size, seed=42):
        """
        randomize the seed, uses a random batch of train_size 
        inputs to train each tree. 
        """
        import random as rand
        rand.seed(seed)
        for i in self.forests:
            newX, newy = self.pull_rand_data(X, y, rand, train_size)
            i.fit(newX, newy, depth)

    def pull_rand_data(self, X, y, rand, train_size):
        newX = []
        newy = []
        for i in range(train_size):
            j = rand.randint(0, len(X)-1)
            newX.append(X[j])
            newy.append(y[j])
        return newX, newy

    
    def predict(self, X):
        """
        Has each forest predict the values
        """
        x = [i.predict(X) for i in self.forests]
        if np.mean(x) <= 0.5:
            return 0
        return 1

