class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        distance_list = []
        for i in self.dataset:
            if self.similarity_function_parameters is not None:
                distance_list.append(self.similarity_function(instance, i, self.similarity_function_parameters))
            else:
                distance_list.append(self.similarity_function(instance, i))

        indexes = sorted(range(len(distance_list)), key=lambda k: distance_list[k])
        distance_list.sort()

        distance_list = distance_list[:self.K]
        indexes = indexes[:self.K]

        labels = [self.dataset_label[j] for j in indexes]

        return max(set(labels), key=labels.count)
