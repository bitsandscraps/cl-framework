import numpy as np

from clfw.mnist import preprocess, PermutedMnist, SplitMnist


def test_preprocess():
    (x_train, y_train), (x_test, y_test) = preprocess()
    assert x_train.ndim == 2 and x_train.shape[-1] == 784
    assert x_test.ndim == 2 and x_test.shape[-1] == 784


def test_permuted_mnist():
    x_train_first = y_train_first = x_test_first = y_test_first = None
    pm = PermutedMnist(10)
    assert cmp_labels_per_task(pm.labels_per_task, [range(10)] * 10)
    for idx, (training_set, test_set) in enumerate(zip(pm.training_sets, pm.test_sets)):
        x_train = training_set.features
        x_test = test_set.features
        y_train = training_set.labels
        y_test = test_set.labels
        assert x_train.ndim == 2
        assert x_test.ndim == 2
        assert y_train.ndim == 1
        assert y_test.ndim == 1
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        if x_train_first is None:
            x_train_first = x_train
            y_train_first = y_train
            x_test_first = x_test
            y_test_first = y_test
        else:
            # Check if the features are actually permuted
            x_train_check = x_train_first == x_train
            y_train_check = y_train_first == y_train
            x_test_check = x_test_first == x_test
            y_test_check = y_test_first == y_test
            assert not np.any(np.all(x_train_check, axis=1))
            assert not np.any(np.all(x_test_check, axis=1))
            assert np.all(y_train_check)
            assert np.all(y_test_check)

            # Check if everything is deep copied
            x_train_orig = x_train.copy()
            y_train_orig = y_train.copy()
            x_test_orig = x_test.copy()
            y_test_orig = y_test.copy()
            x_train_first += 1
            y_train_first += 1
            x_test_first += 1
            y_test_first += 1
            assert np.all(x_train_orig == x_train)
            assert np.all(y_train_orig == y_train)
            assert np.all(x_test_orig == x_test)
            assert np.all(y_test_orig == y_test)
            x_train_first -= 1
            y_train_first -= 1
            x_test_first -= 1
            y_test_first -= 1
    assert idx == 9


def test_split_mnist():
    (_, train), (_, test) = preprocess()
    total_train = train.shape[0]
    total_test = test.shape[0]
    ntrain = ntest = 0
    sm = SplitMnist(2)
    assert cmp_labels_per_task(sm.labels_per_task, [(i, i + 1) for i in range(0, 10, 2)])
    for labels, training_set, test_set in zip(sm.labels_per_task, sm.training_sets, sm.test_sets):
        x_train = training_set.features
        x_test = test_set.features
        y_train = training_set.labels
        y_test = test_set.labels
        assert x_train.ndim == 2
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
    assert ntrain == total_train
    assert ntest == total_test


def cmp_labels_per_task(list1, list2):
    if len(list1) != len(list2):
        return False
    for x, y in zip(list1, list2):
        if len(x) != len(y):
            return False
        for z, w in zip(x, y):
            if z != w:
                return False
    return True