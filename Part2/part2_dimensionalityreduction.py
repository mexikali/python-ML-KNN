import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Load datasets
dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

def visualize_tsne(data, title):
    # Perform t-SNE with hyperparameters
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    embedded_data = tsne.fit_transform(data)

    # Plot t-SNE results
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    plt.title(title)
    plt.show()

def visualize_umap(data, title):
    # Perform UMAP with hyperparameters
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedded_data = umap_model.fit_transform(data)

    # Plot UMAP results
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    plt.title(title)
    plt.show()

# Visualize datasets using t-SNE and UMAP with hyperparameters
visualize_tsne(dataset1, "t-SNE Visualization - Dataset 1")
visualize_umap(dataset1, "UMAP Visualization - Dataset 1")

visualize_tsne(dataset2, "t-SNE Visualization - Dataset 2")
visualize_umap(dataset2, "UMAP Visualization - Dataset 2")
