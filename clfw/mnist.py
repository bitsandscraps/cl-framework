from itertools import zip_longest
from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import mnist

from clfw.core import Array, DataSet, Task, TaskSequence


def preprocess() -> Tuple[DataSet, DataSet]:
    """ Preprocess the MNIST dataset.

    Loads the MNIST dataset and converts each image into a 1D-array.

    Returns:
        A tuple of the training data set and the test data set.
    """
    x_train: Array
    y_train: Array
    x_test: Array
    y_test: Array
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = y_train.astype(np.int_)
    y_test = y_test.astype(np.int_)

    def flatten_each_image(images: Array) -> Array:
        return images.reshape(images.shape[0], -1)

    x_train = flatten_each_image(x_train) / 256
    x_test = flatten_each_image(x_test) / 256
    return DataSet(features=x_train, labels=y_train), DataSet(features=x_test, labels=y_test)


class PermutedMnist(TaskSequence):
    """ Permuted MNIST task sequence

    For each task, every image in the training and test set of MNIST database is
    permuted in a certain random order.
    """
    def __init__(self, ntasks: int = 5) -> None:
        """ Inits a PermutedMnist class with `ntasks` tasks """
        super().__init__(nlabels=10)
        original_training_set, original_test_set = preprocess()

        image_size = original_training_set.features.shape[-1]

        for _ in range(ntasks):
            pattern: np.ndarray = np.random.permutation(image_size)
            # advanced indexing returns a copy
            training_set = DataSet(features=original_training_set.features[..., pattern],
                                   labels=original_training_set.labels.copy())
            test_set = DataSet(features=original_test_set.features[..., pattern],
                               labels=original_test_set.labels.copy())
            self.append(Task(training_set=training_set, test_set=test_set, labels=range(10)))


class SplitMnist(TaskSequence):
    """ Split MNIST task sequence

    Each task has its own labels of interest. The training set and test set of a task
    contains only the images with the corresponding labels.
    """
    def __init__(self, nlabels_per_task: int = 2) -> None:
        """ Inits a SplitMnist class.

        Args:
            nlabels_per_task: number of labels assigned to each task.
        """
        super().__init__(nlabels=10)
        original_training_set, original_test_set = preprocess()

        args = [iter(range(10))] * nlabels_per_task
        labels_of_interest_for_each_task = zip_longest(*args)

        for labels_of_interest in labels_of_interest_for_each_task:
            training_set = self.extract_label(original_training_set, labels_of_interest)
            test_set = self.extract_label(original_test_set, labels_of_interest)
            self.append(Task(training_set=training_set,
                             test_set=test_set,
                             labels=labels_of_interest))

    @staticmethod
    def extract_label(data_set: DataSet, labels_of_interest) -> DataSet:
        """ Extracts images with a label of interest from the data set.

        Returns:
            The data set consisting of every image in the input data set
            with a label of interest.
        """

        # advanced inexing returns a copy
        mask = np.isin(data_set.labels, labels_of_interest)
        return DataSet(features=data_set.features[mask],
                       labels=data_set.labels[mask])
