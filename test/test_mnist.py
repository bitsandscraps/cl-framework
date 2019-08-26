from itertools import zip_longest
from typing import Tuple, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
import tensorflow_datasets as tfds

from clfw import TaskSequence
from clfw.mnist import preprocess, PermutedMnist, SplitMnist

_Dataset = Tuple[np.ndarray, np.ndarray]


def sample_all(
    train: Dataset, valid: Dataset, test: Dataset
) -> Tuple[_Dataset, _Dataset, _Dataset]:
    train_ = tfds.as_numpy(train.batch(50000))
    valid_ = tfds.as_numpy(valid.batch(10000))
    test_ = tfds.as_numpy(test.batch(10000))
    return next(train_), next(valid_), next(test_)


def check_labels_per_task(list1, list2):
    assert len(list1) == len(list2)
    for x, y in zip(list1, list2):
        assert tuple(x) == tuple(y)


def check(dataset: _Dataset, labels: Sequence[int], one_hot: bool) -> int:
    """ Test a dataset.

    :param dataset: dataset under inspection
    :param labels: labels of interest
    :param one_hot: whether or not the label is one-hot encoded
    :return: number of samples in the dataset
    """
    x, y = dataset
    assert x.ndim == 2
    assert np.max(x) == 255 / 256
    if one_hot:
        assert y.ndim == 2
        assert y.shape[-1] == 10
        y = np.argmax(y, axis=-1)
    else:
        assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    counts = np.histogram(y, range(11))[0]
    for idx in range(10):
        if idx in labels:
            assert counts[idx] > 0
        else:
            assert counts[idx] == 0
    return y.shape[0]


def check_length(model: TaskSequence, length: int) -> None:
    assert len(model.training_sets) == length
    assert len(model.validation_sets) == length
    assert len(model.test_sets) == length


def test_preprocess() -> None:
    datasets, image_size = preprocess()
    assert image_size == 784

    def sample(ds) -> Tuple[tf.Tensor, tf.Tensor]:
        return next(iter(ds.cache().batch(1)))

    for ds in datasets:
        assert sample(ds)[0].shape[1] == 784


def _test_permuted_mnist(one_hot: bool) -> None:
    train_first = valid_first = test_first = None
    pm = PermutedMnist(10, one_hot=one_hot)
    check_labels_per_task(pm.labels_per_task, [range(10)] * 10)
    for label, train, valid, test in zip(pm.labels_per_task, pm.training_sets,
                                         pm.validation_sets, pm.test_sets):
        train, valid, test = sample_all(train, valid, test)
        label = tuple(label)
        check(train, label, one_hot)
        check(valid, label, False)
        check(test, label, False)

        if train_first is None:
            train_first, valid_first, test_first = train, valid, test
        else:
            assert not np.allclose(train[0], train_first[0])
            assert not np.allclose(valid[0], valid_first[0])
            assert not np.allclose(test[0], test_first[0])
            assert np.all(train_first[1] == train[1])
            assert np.all(valid_first[1] == valid[1])
            assert np.all(test_first[1] == test[1])
    check_length(pm, 10)


def _test_split_mnist(one_hot: bool):
    sm = SplitMnist(2, one_hot=one_hot)
    check_labels_per_task(sm.labels_per_task,
                          [(i, i + 1) for i in range(0, 10, 2)])
    ntrain = nvalid = ntest = 0
    for label, train, valid, test in zip(
            sm.labels_per_task, sm.training_sets, sm.validation_sets, sm.test_sets):
        train, valid, test = sample_all(train, valid, test)
        label = tuple(label)
        ntrain += check(train, label, one_hot)
        nvalid += check(valid, label, False)
        ntest += check(test, label, False)

    check_length(sm, 5)
    assert ntrain == 50000
    assert nvalid == 10000
    assert ntest == 10000


def test_permuted_mnist():
    _test_permuted_mnist(True)
    _test_permuted_mnist(False)


def test_split_mnist():
    _test_split_mnist(True)
    _test_split_mnist(False)


if __name__ == '__main__':
    test_permuted_mnist()
