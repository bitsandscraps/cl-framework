from typing import Optional
import numpy as np

from clfw import Array, DataSet, Model, Task, TaskSequence


class ToyModel(Model):
    def __init__(self):
        self.memory_features: Optional[Array] = None
        self.memory_labels: Optional[Array] = None

    def train(self, training_set: DataSet) -> None:
        if self.memory_features is None:
            self.memory_features = training_set.features
            self.memory_labels = training_set.labels
        else:
            self.memory_features = np.vstack([training_set.features, self.memory_features])
            self.memory_labels = np.hstack([training_set.labels, self.memory_labels])

    def classify(self, features: Array) -> Array:
        if self.memory_features is None:
            return np.zeros(features.shape[0])
        prediction = np.empty(features.shape[0])
        for idx, feature in enumerate(features):
            search: Array = np.all(self.memory_features == feature, axis=1).nonzero()
            if search[0].size == 0:
                prediction[idx] = 0     # default prediction
            else:
                prediction[idx] = self.memory_labels[search[0][0]]
        return prediction


TOY_TASK_1 = Task(training_set=DataSet(features=np.arange(8).reshape(4, 2),
                                       labels=np.arange(4)),
                  test_set=DataSet(features=np.asarray([[0, 1], [2, 3], [5, 6], [7, 8]]),
                                   labels=np.asarray([0, 1, 5, 0])),
                  labels=range(6))

TOY_TASK_2 = Task(training_set=DataSet(features=np.arange(8).reshape(4, 2) + 1,
                                       labels=np.arange(4) + 1),
                  test_set=DataSet(features=TOY_TASK_1.test_set.features + 1,
                                   labels=TOY_TASK_1.test_set.labels + 1),
                  labels=range(7))


def test_evaluate_model():
    model = ToyModel()
    assert model.evaluate(TOY_TASK_1.test_set) == (2, 4)
    model.train(TOY_TASK_1.training_set)
    assert model.evaluate(TOY_TASK_1.test_set) == (3, 4)
    model = ToyModel()
    assert model.evaluate(TOY_TASK_2.test_set) == (0, 4)
    model.train(TOY_TASK_2.training_set)
    assert model.evaluate(TOY_TASK_2.test_set) == (2, 4)


def test_task_sequence():
    task_seq = TaskSequence(10, [TOY_TASK_1, TOY_TASK_2])
    assert task_seq.ntasks == 2
    assert task_seq.nlabels == 10
    assert task_seq.feature_dim == (2,)
    acc, accmat = task_seq.evaluate(ToyModel())
    assert np.all(acc == np.asarray([0.25, 0.375, 0.5]))
    assert np.all(accmat == np.asarray([[0.5, 0], [0.75, 0], [0.5, 0.5]]))
