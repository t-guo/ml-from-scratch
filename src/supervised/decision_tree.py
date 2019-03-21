import numpy as np
from src.utils import calculations, validations


class DecisionTree(object):
    def __init__(self, min_sample=1, max_depth=None):
        self.min_sample = min_sample
        self.max_depth = max_depth

    def fit(self, X, y):
        pass

    def predict(self, x_pred):
        pass
