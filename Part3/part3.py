import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


dataset = pickle.load(open("../data/part3_dataset.data", "rb"))

linkage_criteria = ['single', 'complete']
distance_measures = ['euclidean', 'cosine']
best_configuration = None
num_of_clusters = [2, 3, 4, 5]

for K in num_of_clusters:
    max_avg_silhouette = -1
    for linkage_criterion in linkage_criteria:
        for distance_measure in distance_measures:
            model = AgglomerativeClustering(K, linkage=linkage_criterion, affinity=distance_measure)
            clusters = model.fit_predict(dataset)

            # Dendrogram Plot
            linkage_matrix = linkage(dataset, method=linkage_criterion, metric=distance_measure)
            dendrogram(linkage_matrix)
            plt.title(f'Dendrogram ({linkage_criterion}, {distance_measure}, {K})')
            plt.show()

            # Silhouette Analysis
            silhouette_avg = silhouette_score(dataset, clusters)
            print(f'Average Silhouette Score ({linkage_criterion}, {distance_measure}, {K}): {silhouette_avg}')

            # Update best configuration
            if silhouette_avg > max_avg_silhouette:
                max_avg_silhouette = silhouette_avg
                best_configuration = (linkage_criterion, distance_measure, K)

    print(f'The best configuration is: {best_configuration} with an average silhouette score of {max_avg_silhouette}')
