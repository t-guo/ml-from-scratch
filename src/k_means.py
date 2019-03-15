import numpy as np
from src.utils import calculations


class KMeans(object):
    """
    KMeans clustering
    """
    def __init__(self, k=2, init='uniform', stopping_distance=0, max_iterations=500):
        if not k > 0:
            raise Exception('k needs to be greater than 0.')
        if not stopping_distance >= 0:
            raise Exception('stopping_distance needs to be greater than or equal to 0.')
        if not max_iterations > 0:
            raise Exception('max_iterations needs to be greater than 0.')
        if init not in ['uniform', 'k++']:
            raise Exception('init needs to be one of [\'uniform\', \'k++\'].')

        self.k = k
        self.init = init
        self.max_iterations = max_iterations
        self.stopping_distance = stopping_distance
        self.centroids = None

    def _init_random_centroids(self, X):
        n, m = np.shape(X)
        self.centroids = np.zeros((self.k, m))

        if self.init == 'uniform':
            # Choose k centers uniformly at random from among the data points
            self.centroids = X[np.random.choice(range(n), self.k, replace=False)]
        elif self.init == 'k++':
            # Randomly pick a center
            self.centroids[0] = X[np.random.choice(range(n), 1)]

            # Computer distance between center for each
            for i in range(1, self.k):
                distances = []
                # Find distance between each point to nearest center
                for x in X:
                    closest_dist = float('inf')
                    for j in range(i):
                        distance = calculations.euclidean_distance(x, self.centroids[j])
                        if distance < closest_dist:
                            closest_dist = distance
                    distances.append(closest_dist)

                # Re-sample next center with probabilities proportional to shortest distance
                self.centroids[i] = X[np.random.choice(range(n), 1, p=[d/sum(distances) for d in distances])]

    def _closest_centroid(self, sample):
        """ Return the index of the closest centroid to the sample """
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(self.centroids):
            distance = calculations.euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def fit(self, X):
        """
        Do K-means clustering
        :param X: np array of features
        :return: cluster indexes and centroids of clusters
        """
        # Initialize centroid, iteration number, and difference
        self._init_random_centroids(X)
        iter = 0
        diff = float('inf')
        n, m = np.shape(X)

        while iter <= self.max_iterations and diff > self.stopping_distance:
            # Assign samples to closet centroids
            cluster_labels = []

            for i, x in enumerate(X):
                centroid_i = self._closest_centroid(x)
                cluster_labels.append(centroid_i)

            unique_centroids, counts = np.unique(cluster_labels, return_counts=True)

            if len(unique_centroids) != len(self.centroids):
                raise Exception('algorithm not converging, some clusters have no members.')

            # Calculate new centroids and difference between previous centroids
            new_centroids = np.zeros((self.k, m))

            for j, centroid in enumerate(self.centroids):
                new_centroids[j] = np.mean(X[np.array(cluster_labels) == j], axis=0)
            diff = np.sum(new_centroids - self.centroids)

            self.centroids = new_centroids

            iter += 1

        return cluster_labels
