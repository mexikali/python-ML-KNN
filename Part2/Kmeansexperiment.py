from Part2.Kmeans import KMeans
import pickle
import numpy as np
import matplotlib.pyplot as plt

def runKmeans(dataset, num_runs=10):
    avg_losses = []

    for k in range(2, 11):  # K values from 2 to 10
        k_losses = []

        for _ in range(num_runs):
            # Run the algorithm num_runs times and pick the lowest loss value
            _, _, loss = KMeans(dataset, K=k).run()
            k_losses.append(loss)

        avg_losses.append(np.min(k_losses))

    return avg_losses

def plot_elbow_method(algorithm_name, avg_losses, dataset_name=""):
    plt.plot(range(2, 11), avg_losses, marker='o')
    plt.title(f'Elbow Method for {algorithm_name} on {dataset_name}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Average Loss')
    plt.show()



dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))

# Run KMeans and plot the elbow method
kmeans_avg_losses_1 = runKmeans(dataset1)
plot_elbow_method("KMeans", kmeans_avg_losses_1, "Dataset 1")

kmeans_avg_losses_2 = runKmeans(dataset2)
plot_elbow_method("KMeans", kmeans_avg_losses_2, "Dataset 2")
