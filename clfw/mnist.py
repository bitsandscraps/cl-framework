from functools import reduce
from itertools import zip_longest
import operator
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetInfo

from clfw import Task, TaskSequence


def preprocess() -> Tuple[Dataset, Dataset, int]:
    """ Preprocess the MNIST dataset.

    Load it. Normalize it. Flatten it.

    :return: training dataset, test dataset, image_ size
    """
    dataset_info: DatasetInfo
    (train, test), dataset_info = tfds.load(
        'mnist', as_supervised=True, with_info=True, split=['train', 'test'],
        as_dataset_kwargs={'shuffle_files': False})
    image: tfds.features.Image = dataset_info.features['image']
    image_size = reduce(operator.mul, image.shape)

    def normalize_and_flatten(image_: tf.Tensor, label: tf.Tensor):
        image_: tf.Tensor = tf.cast(image_, tf.float32) / 256
        return tf.reshape(image_, (image_size,)), label

    train = train.map(normalize_and_flatten)
    test = test.map(normalize_and_flatten)

    return train, test, image_size


class PermutedMnist(TaskSequence):
    """ Permuted MNIST task sequence

    For each task, every image in the training and test set of MNIST database is
    permuted in a certain random order.
    """
    def __init__(self, ntasks: int = 5) -> None:
        """ Inits a PermutedMnist class with `ntasks` tasks """
        super().__init__(nlabels=10)
        original_train, original_test, image_size = preprocess()

        for _ in range(ntasks):
            pattern = np.random.permutation(image_size)

            def permute(image: tf.Tensor, label: tf.Tensor):
                return tf.gather(image, pattern), label

            # advanced indexing returns a copy
            train = original_train.map(permute)
            test = original_test.map(permute)
            self.append(Task(train=train, test=test, labels=range(10)))


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
        original_train, original_test, image_size = preprocess()

        args = [iter(range(10))] * nlabels_per_task
        labels_of_interest_for_each_task = zip_longest(*args)

        for labels_of_interest in labels_of_interest_for_each_task:

            def check_label(image: tf.Tensor, label:tf.Tensor):
                del image
                first = True
                result: tf.Tensor
                for lab in labels_of_interest:
                    check = tf.equal(label, lab)
                    if first:
                        result = check
                        first = False
                    else:
                        result = tf.math.logical_or(result, check)
                return result

            train = original_train.filter(check_label)
            test = original_test.filter(check_label)
            self.append(Task(train=train, test=test, labels=labels_of_interest))
