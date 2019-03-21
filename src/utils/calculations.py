import math
import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes the l2 distance between two vectors
    """
    distance = 0
    # Squared distance between each coordinate
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


def euclidean_distance_matrix(X, x):
    """
    Compute l2 distance between a vector and to an array of vectors
    """
    distance_matrix = np.sqrt(np.sum((X - x) ** 2, axis=1))
    return distance_matrix

