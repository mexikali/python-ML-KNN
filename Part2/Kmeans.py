import numpy as np
from Distance import Distance

class KMeans:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        self.clusters = {i: [] for i in range(K)}
        self.cluster_centers = self.initialize_centers()

    def initialize_centers(self):
        # Initialize cluster centers by randomly selecting K data points
        indices = np.random.choice(len(self.dataset), self.K, replace=False)
        centers = {i: self.dataset[index] for i, index in enumerate(indices)}
        return centers

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        loss = 0
        for cluster_index in range(self.K):
            center = self.cluster_centers[cluster_index]
            cluster_points = self.clusters[cluster_index]
            for point in cluster_points:
                loss += Distance.calculateMinkowskiDistance(point, center) ** 2
        return loss

    def assign_points_to_clusters(self):
        """Assign each data point to the nearest cluster"""
        self.clusters = {i: [] for i in range(self.K)}
        for point in self.dataset:
            distances = [Distance.calculateMinkowskiDistance(point, center) for center in self.cluster_centers.values()]
            nearest_cluster = np.argmin(distances)
            self.clusters[nearest_cluster].append(point)

    def update_cluster_centers(self):
        """Update cluster centers based on the mean of data points in each cluster"""
        for cluster_index in range(self.K):
            cluster_points = self.clusters[cluster_index]
            if len(cluster_points) > 0:
                self.cluster_centers[cluster_index] = np.mean(cluster_points, axis=0)

    def run(self, max_iterations=100, tol=1e-4):
        """Kmeans algorithm implementation"""
        for _ in range(max_iterations):
            old_centers = dict(self.cluster_centers)
            self.assign_points_to_clusters()
            self.update_cluster_centers()

            # Check convergence
            if all(np.sum((old_centers[i] - self.cluster_centers[i]) ** 2) < tol for i in range(self.K)):
                break

        return self.cluster_centers, self.clusters, self.calculateLoss()
