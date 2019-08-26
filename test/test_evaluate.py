from typing import Iterable, Optional, Tuple

import numpy as np
from tensorflow.python.data import Dataset
import tensorflow_datasets as tfds

from clfw import Array, Model, Task, TaskSequence


TOY_TASK_1_TRAIN = (np.arange(8).reshape(4, 2), np.arange(4))
TOY_TASK_1_TEST = (np.asarray([[0, 1], [2, 3], [5, 6], [7, 8]]),
                   np.asarray([0, 1, 5, 0]))
TOY_TASK_2_TRAIN = (np.arange(8).reshape(4, 2) + 1, np.arange(4) + 1)
TOY_TASK_2_TEST = (TOY_TASK_1_TEST[0] + 1, TOY_TASK_1_TEST[1] + 1)


class ToyModel(Model):
    def __init__(self):
        self.memory_features: Optional[Array] = None
        self.memory_labels: Optional[Array] = None

    def train(self,
              training_set: Dataset,
              validation_set: Dataset,
              labels: Iterable[int]) -> None:
        del validation_set
        del labels

        features, labels = next(tfds.as_numpy(training_set.batch(1000)))

        if self.memory_features is None:
            self.memory_features, self.memory_labels = features, labels
        else:
            self.memory_features = np.vstack([features, self.memory_features])
            self.memory_labels = np.hstack([labels, self.memory_labels])

    def evaluate(self, test_set: Dataset) -> Tuple[int, int]:
        features, labels = next(tfds.as_numpy(test_set.batch(1000)))
        ntotal: int = labels.shape[0]
        if self.memory_features is None:
            return int(np.sum(labels == 0)), ntotal
        prediction = np.empty(features.shape[0])
        for idx, feature in enumerate(features):
            search: Array = np.all(self.memory_features == feature, axis=1).nonzero()
            if search[0].size == 0:
                prediction[idx] = 0     # default prediction
            else:
                prediction[idx] = self.memory_labels[search[0][0]]
        return int(np.sum(prediction == labels)), ntotal


def toy_task_1():
    return Task(train=Dataset.from_tensor_slices(TOY_TASK_1_TRAIN),
                valid=None,
                test=Dataset.from_tensor_slices(TOY_TASK_1_TEST),
                labels=range(7))


def toy_task_2():
    return Task(train=Dataset.from_tensor_slices(TOY_TASK_2_TRAIN),
                valid=None,
                test=Dataset.from_tensor_slices(TOY_TASK_2_TEST),
                labels=range(7))


def test_evaluate_model():
    model = ToyModel()
    t1 = toy_task_1()
    t2 = toy_task_2()
    assert model.evaluate(t1.test) == (2, 4)
    model.train(t1.train, t1.valid, t1.labels)
    assert model.evaluate(t1.test) == (3, 4)
    model = ToyModel()
    assert model.evaluate(t2.test) == (0, 4)
    model.train(t2.train, t2.valid, t2.labels)
    assert model.evaluate(t2.test) == (2, 4)


def test_task_sequence():
    task_seq = TaskSequence(10, False, [toy_task_1(), toy_task_2()])
    assert task_seq.ntasks == 2
    assert task_seq.nlabels == 10
    assert task_seq.feature_dim == [2]
    acc, accmat = task_seq.evaluate(ToyModel())
    assert np.all(acc == np.asarray([0.25, 0.375, 0.5]))
    assert np.all(accmat == np.asarray([[0.5, 0], [0.75, 0], [0.5, 0.5]]))


def check_one_hot(dataset: Optional[Dataset], nlabels: int):
    if dataset is None:
        return
    _, labels = next(tfds.as_numpy(dataset.batch(1000)))
    assert labels.shape[1] == nlabels


def test_onehot():
    task_seq = TaskSequence(10, True, [toy_task_1(), toy_task_2()])
    assert task_seq.ntasks == 2
    assert task_seq.nlabels == 10
    assert task_seq.feature_dim == [2]
    for train, valid, test in zip(task_seq.training_sets,
                                  task_seq.validation_sets,
                                  task_seq.test_sets):
        check_one_hot(train, 10)
        check_one_hot(valid, 10)
        check_one_hot(test, 10)

