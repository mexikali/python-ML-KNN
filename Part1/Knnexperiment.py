import pickle
from Distance import Distance
from Knn import KNN
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load the dataset
dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))

# hyperparameter configurations to test
k_values = [1, 3, 5, 7, 9]
metric_values = ['cosine', 'euclidean']

# Number of cross-validation folds
num_folds = 10

# Number of repetitions
num_repeats = 5

# Numpy List for accuracy scores
accuracy_scores = np.zeros((num_repeats, len(k_values), len(metric_values), num_folds))

# Repeating the cross-validation procedure multiple times
for repeat in range(num_repeats):
    # Shuffle the dataset
    shuffled_indices = np.random.permutation(len(dataset))
    shuffled_dataset = dataset[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    # Initializing stratified k-fold cross-validator
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=repeat)

    for k_index, k in enumerate(k_values):
        for metric_index, metric in enumerate(metric_values):
            for fold_index, (train_index, test_index) in enumerate(skf.split(shuffled_dataset, shuffled_labels)):
                # Split the dataset into training and testing sets
                train_data, test_data = shuffled_dataset[train_index], shuffled_dataset[test_index]
                train_labels, test_labels = shuffled_labels[train_index], shuffled_labels[test_index]

                # Initializing the KNN model according to metric
                if metric_index == 0:
                    knn = KNN(dataset=train_data, data_label=train_labels, similarity_function=Distance.calculateCosineDistance, K=k)
                else:
                    knn = KNN(dataset=train_data, data_label=train_labels, similarity_function=Distance.calculateMinkowskiDistance, K=k)

                # Predict using the test set
                predictions = [knn.predict(instance) for instance in test_data]

                # Calculate and store accuracy score
                accuracy_scores[repeat, k_index, metric_index, fold_index] = accuracy_score(test_labels, predictions)

# Compute mean and confidence intervals for each hyperparameter configuration
for k_index, k in enumerate(k_values):
    for metric_index, metric in enumerate(metric_values):
        mean_accuracy = np.mean(accuracy_scores[:, k_index, metric_index, :], axis=1)
        confidence_intervals = np.percentile(accuracy_scores[:, k_index, metric_index, :], [2.5, 97.5], axis=1)

        # Print results for the current hyperparameter combination
        print(f"\nResults for k = {k}, metric = {metric}:")
        print(f"Mean Accuracy: {mean_accuracy}")
        print(f"95% Confidence Intervals: {confidence_intervals}")
