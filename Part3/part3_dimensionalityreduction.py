import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA


dataset = pickle.load(open("../data/part3_dataset.data", "rb"))

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(dataset)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], label='t-SNE')
plt.title('t-SNE Visualization')
plt.show()

# UMAP
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2)
umap_result = umap_model.fit_transform(dataset)
plt.scatter(umap_result[:, 0], umap_result[:, 1], label='UMAP')
plt.title('UMAP Visualization')
plt.show()

# PCA (for comparison)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(dataset)
plt.scatter(pca_result[:, 0], pca_result[:, 1], label='PCA')
plt.title('PCA Visualization')
plt.show()

