from abc import ABC, abstractmethod
from typing import Iterable, List, NamedTuple, Optional, Tuple

import numpy as np

Array = np.ndarray


class DataSet(NamedTuple):
    features: Array
    labels: Array


class Task(NamedTuple):
    training_set: DataSet
    test_set: DataSet
    labels: Iterable[int]


class Model(ABC):
    """ Base class for a model for continual learning """
    @abstractmethod
    def train(self, training_set: DataSet, labels: Iterable[int]) -> None:
        pass

    @abstractmethod
    def classify(self, features: Array) -> Array:
        pass

    def evaluate(self, test_set: DataSet) -> Tuple[int, int]:
        """ Evaluate the model using the given test set.

        Returns:
            1. number of correct predictions
            2. number of examples in the test set

            For example: 948, 1000

            This means the test set has 1000 examples and the model has achieved 94.8%
            accuracy on it.
        """
        prediction: Array = self.classify(test_set.features).ravel()
        labels: Array = test_set.labels.ravel()
        ncorrect = np.count_nonzero(labels == prediction)
        ntotal = labels.size
        return ncorrect, ntotal


class TaskSequence:
    """ Sequence of tasks to test a continual learning algorithm.

    Attributes:
        feature_dim: the dimension of a feature
        labels_per_task: list of labels each task's training set contains
        nlabels: number of total possible labels
        ntasks: total number of tasks
        training_sets: list of training sets
        test_sets: list of test sets
    """
    def __init__(self, nlabels: int, tasks: Optional[Iterable[Task]] = None) -> None:
        self.labels_per_task: List[Iterable[int]] = []
        self.nlabels = nlabels
        self.ntasks: int = len(tasks) if tasks is not None else 0
        self.training_sets: List[DataSet] = []
        self.test_sets: List[DataSet] = []
        if tasks:
            for task in tasks:
                self.training_sets.append(task.training_set)
                self.test_sets.append(task.test_set)
                self.labels_per_task.append(task.labels)

    @property
    def feature_dim(self) -> Tuple[int, ...]:
        if not self.training_sets:
            raise ValueError("There are no tasks yet.")
        # 0-th element is the number of training examples
        return self.training_sets[0].features.shape[1:]

    def append(self, task: Task) -> None:
        """ Append a training set test set pair to the sequence. """
        self.ntasks += 1
        self.training_sets.append(task.training_set)
        self.test_sets.append(task.test_set)
        self.labels_per_task.append(task.labels)

    def evaluate(self, model: Model) -> Tuple[Array]:
        """ Evaluate the model using the given sequence of tasks.

        Returns:
            1. average_accuracy measured after learning each task
            2. accuracy per task measured after learning each task

            Assume there are N tasks.
            1 is a length N + 1 vector whose i-th element is the average accuracy on the whole
            test set after training up to task i - 1.
            2 is a (N + 1) x N matrix whose (i, j)-th element is the accuracy on test set
            of task j after training up to task i - 1.
        """

        accuracy_matrix = np.empty((self.ntasks + 1, self.ntasks))
        average_accuracy = np.empty((self.ntasks + 1,))
        ncorrect = ntotal = 0
        for test_idx, test_set in enumerate(self.test_sets):
            ncorrect_task, ntotal_task = model.evaluate(test_set)
            accuracy_matrix[0, test_idx] = ncorrect_task / ntotal_task
            ncorrect += ncorrect_task
            ntotal += ntotal_task
        average_accuracy[0] = ncorrect / ntotal
        for train_idx, training_set in enumerate(self.training_sets):
            model.train(training_set)
            ncorrect = ntotal = 0
            for test_idx, test_set in enumerate(self.test_sets):
                ncorrect_task, ntotal_task = model.evaluate(test_set)
                accuracy_matrix[train_idx + 1, test_idx] = ncorrect_task / ntotal_task
                ncorrect += ncorrect_task
                ntotal += ntotal_task
            average_accuracy[train_idx + 1] = ncorrect / ntotal
        return average_accuracy, accuracy_matrix
