import numpy as np
from sklearn.metrics.pairwise import cosine_distances

class KMedoids:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        initial_medoid_indices = np.random.choice(len(self.dataset), self.K, replace=False)
        self.cluster_medoids = {i: self.dataset[idx] for i, idx in enumerate(initial_medoid_indices)}
        self.clusters = {i: [] for i in range(K)}

    def calculate_loss(self):
        """Loss function implementation of Equation 2"""
        loss = 0
        for cluster_index in range(self.K):
            medoid = self.cluster_medoids[cluster_index]
            cluster_points = self.clusters[cluster_index]
            loss += np.sum(cosine_distances([medoid], cluster_points))
        return loss

    def assign_points_to_clusters(self):
        """Assign each data point to the nearest cluster medoid"""
        self.clusters = {i: [] for i in range(self.K)}
        distances = cosine_distances(self.dataset, list(self.cluster_medoids.values()))
        cluster_assignments = np.argmin(distances, axis=1)

        for i, point in enumerate(self.dataset):
            nearest_cluster = cluster_assignments[i]
            self.clusters[nearest_cluster].append(point)

    def update_cluster_medoids(self):
        """Update cluster medoids by selecting the point with the minimum total distance to other points in the cluster"""
        for cluster_index in range(self.K):
            cluster_points = self.clusters[cluster_index]
            if cluster_points:
                distances = np.sum(cosine_distances([point], cluster_points) for point in cluster_points)
                best_medoid_index = np.argmin(distances)
                self.cluster_medoids[cluster_index] = cluster_points[best_medoid_index]

    def run(self, max_iterations=100, tol=1e-4):
        """KMedoids algorithm implementation"""
        for _ in range(max_iterations):
            old_medoids = dict(self.cluster_medoids)
            self.assign_points_to_clusters()
            self.update_cluster_medoids()

            # Check convergence for each cluster individually
            if all(
                np.sum(cosine_distances([old_medoids[i]], [self.cluster_medoids[i]])) < tol
                for i in range(self.K)
            ):
                break

        return self.cluster_medoids, self.clusters, self.calculate_loss()
