import numpy as np
from src.utils import calculations, validations
from collections import Counter


class KNN(object):
    """
    KNN Base
    """
    def __init__(self, k, weights='uniform'):
        if not k > 0:
            raise Exception('k needs to be greater than 0.')
        if weights not in ['uniform', 'weighted']:
            raise Exception('init needs to be one of [\'uniform\', \'weighted\'].')

        self.k = k
        self.weights = weights
        self.X = None
        self.y = None

    def fit(self, X, y):
        pass

    def predict(self, x_pred):
        pass


class KNNClassifier(KNN):
    """
    KNN Classification
    """
    def fit(self, X, y):
        validations.is_classification(y)

        self.X = X
        self.y = y

    def predict(self, x_pred):
        validations.is_fitted(self, 'X')
        y_pred = []

        for x in x_pred:
            distance_matrix = calculations.euclidean_distance_matrix(self.X, x)
            distance_indexes = np.argsort(distance_matrix)[0:self.k]
            distances = distance_matrix[distance_indexes]

            if self.weights == 'uniform':
                y_pred.append(Counter(y[distance_indexes]).most_common(1)[0][0])
            elif self.weights == 'weighted':
                c = Counter()
                for k, v in zip(y[distance_indexes], distances):
                    c.update({k: v})

                y_pred.append(c.most_common(1)[0][0])

        return y_pred


class KNNRegressor(KNN):
    """
    KNN Regression
    """
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x_pred):
        validations.is_fitted(self, 'X')
        y_pred = []

        for x in x_pred:
            distance_matrix = calculations.euclidean_distance_matrix(self.X, x)
            distance_indexes = np.argsort(distance_matrix)[0:self.k]
            distances = distance_matrix[distance_indexes]

            if self.weights == 'uniform':
                y_pred.append(np.average(self.y[distance_indexes]))
            elif self.weights == 'weighted':
                y_pred.append(np.average(self.y[distance_indexes], weights=distances))

        return y_pred
