import numpy as np
from src.utils import calculations, validations


class KNNClassifier(object):
    def __init__(self, k, weights='weighted'):
        self.k = k
        self.weights = weights
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_pred):

        validations.is_fitted(self, 'X')
        y_pred = []

        for x in X_pred:
            distance_matrix = calculations.euclidean_distance_matrix(self.X)
            distances = np.argsort(distance_matrix)[0:self.k]

            print y[distances]
            y_pred.append(np.mode())

        return y_pred



