from itertools import zip_longest
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from clfw.mnist import preprocess, PermutedMnist, SplitMnist


def test_preprocess():
    train, test, image_size = preprocess()
    assert image_size == 784

    def sample(ds) -> Tuple[tf.Tensor, tf.Tensor]:
        return next(iter(ds.cache().batch(1)))

    assert sample(train)[0].shape[1] == 784
    assert sample(test)[0].shape[1] == 784


def test_permuted_mnist():
    x_train_first = y_train_first = x_test_first = y_test_first = None
    pm = PermutedMnist(10)
    assert cmp_labels_per_task(pm.labels_per_task, [range(10)] * 10)
    for idx, (trains, tests) in enumerate(zip(pm.training_sets, pm.test_sets)):
        trains = tfds.as_numpy(trains.batch(60000))
        tests = tfds.as_numpy(tests.batch(10000))
        x_train, y_train = next(trains)
        x_test, y_test = next(tests)
        assert x_train.ndim == 2
        assert y_train.ndim == 1
        assert np.max(x_train) == 255/256
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        if x_train_first is None:
            x_train_first = x_train
            y_train_first = y_train
            x_test_first = x_test
            y_test_first = y_test
        else:
            assert not np.allclose(x_train, x_train_first)
            assert not np.allclose(x_test, x_test_first)
            assert np.all(y_train_first == y_train)
            assert np.all(y_test_first == y_test)
    assert idx == 9


def test_split_mnist():
    sm = SplitMnist(2)
    assert cmp_labels_per_task(sm.labels_per_task, [(i, i + 1) for i in range(0, 10, 2)])
    ntrain = 0
    ntest = 0
    for labels, trains, tests in zip(sm.labels_per_task, sm.training_sets, sm.test_sets):
        trains = tfds.as_numpy(trains.batch(60000))
        tests = tfds.as_numpy(tests.batch(10000))
        x_train, y_train = next(trains)
        x_test, y_test = next(tests)
        assert x_train.ndim == 2
        assert np.max(x_train) == 255/256
        assert x_test.ndim == 2
        assert y_train.ndim == 1
        assert y_test.ndim == 1
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        ntrain += y_train.shape[0]
        ntest += y_test.shape[0]
        for y in y_train:
            assert y in labels
        for y in y_test:
            assert y in labels
    assert len(sm.training_sets) == 5
    assert len(sm.test_sets) == 5
    assert ntrain == 60000
    assert ntest == 10000


def cmp_labels_per_task(list1, list2):
    if len(list1) != len(list2):
        return False
    for x, y in zip(list1, list2):
        if tuple(x) != tuple(y):
            return False
    return True


if __name__ == '__main__':
    test_permuted_mnist()
