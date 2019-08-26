from functools import reduce
from itertools import zip_longest
import operator
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetInfo

from clfw import Task, TaskSequence

TRAIN_SET_SIZE = 50000


def preprocess() -> Tuple[Tuple[Dataset, Dataset, Dataset], int]:
    """ Preprocess the MNIST dataset.

    Load it. Normalize it. Flatten it.

    :return: training dataset, validation dataset, test dataset, image_ size
    """
    dataset_info: DatasetInfo
    datasets, dataset_info = tfds.load(
        'mnist:3.*.*', as_supervised=True, with_info=True,
        split=[f'train[:{TRAIN_SET_SIZE}]', f'train[{TRAIN_SET_SIZE}:]', 'test'],
        as_dataset_kwargs={'shuffle_files': False})
    image: tfds.features.Image = dataset_info.features['image']
    image_size = reduce(operator.mul, image.shape)

    def normalize_and_flatten(image_: tf.Tensor, label: tf.Tensor):
        image_: tf.Tensor = tf.cast(image_, tf.float32) / 256
        return tf.reshape(image_, (image_size,)), label

    train, valid, test = (ds.map(normalize_and_flatten) for ds in datasets)

    return (train, valid, test), image_size


class PermutedMnist(TaskSequence):
    """ Permuted MNIST task sequence

    For each task, every image in the training and test set of MNIST database is
    permuted in a certain random order.
    """
    def __init__(self, ntasks: int = 5, one_hot: bool = True) -> None:
        """ Inits a PermutedMnist class with `ntasks` tasks """
        super().__init__(nlabels=10, one_hot=one_hot)
        datasets, image_size = preprocess()

        for _ in range(ntasks):
            pattern = np.random.permutation(image_size)

            def permute(image: tf.Tensor, label: tf.Tensor):
                return tf.gather(image, pattern), label

            train, valid, test = (ds.map(permute) for ds in datasets)
            self.append(Task(train=train, valid=valid,
                             test=test, labels=range(10)))


class SplitMnist(TaskSequence):
    """ Split MNIST task sequence

    Each task has its own labels of interest. The training set and test set of a task
    contains only the images with the corresponding labels.
    """
    def __init__(self, nlabels_per_task: int = 2, one_hot: bool = True) -> None:
        """ Inits a SplitMnist class.

        Args:
            nlabels_per_task: number of labels assigned to each task.
        """
        super().__init__(nlabels=10, one_hot=one_hot)
        datasets, _ = preprocess()

        args = [iter(range(10))] * nlabels_per_task
        labels_of_interest_for_each_task = zip_longest(*args)

        for labels_of_interest in labels_of_interest_for_each_task:

            def check_label(image: tf.Tensor, label: tf.Tensor):
                del image
                first = True
                result: Optional[tf.Tensor] = None
                for lab in labels_of_interest:
                    check = tf.equal(label, lab)
                    if first:
                        result = check
                        first = False
                    else:
                        result = tf.math.logical_or(result, check)
                return result

            train, valid, test = (ds.filter(check_label) for ds in datasets)
            self.append(Task(train=train, valid=valid,
                             test=test, labels=labels_of_interest))
