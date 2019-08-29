from abc import ABC, abstractmethod
import os.path
from typing import Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset

from clfw.util import save_results

Array = np.ndarray


class Task(NamedTuple):
    train: Dataset
    valid: Optional[Dataset]
    test: Dataset
    labels: Iterable[int]


def to_one_hot(task: Task, nlabels: int) -> Task:
    train, valid, test = (task.train, task.valid, task.test)

    def one_hot(feature: tf.Tensor, label: tf.Tensor):
        return feature, tf.one_hot(label, nlabels)
    train = train.map(one_hot)
    return Task(train=train, valid=valid, test=test, labels=task.labels)


def evaluate(model: "Model", evaluation_sets: List[Dataset]) -> List[float]:
    return [model.evaluate(eval_set) for eval_set in evaluation_sets]


class Model(ABC):
    """ Base class for a model for continual learning """
    @abstractmethod
    def train(self,
              training_set: Dataset,
              validation_set: Dataset,
              labels: Iterable[int]) -> None:
        pass

    @abstractmethod
    def evaluate(self, test_set: Dataset) -> float:
        """ Evaluate the model on a test set.

        :return: Accuracy. Between 0 and 1.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """ Reset the model to its initial state. """
        pass


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
    def __init__(self, nlabels: int, one_hot: bool = True,
                 tasks: Optional[Iterable[Task]] = None) -> None:
        self.labels_per_task: List[Iterable[int]] = []
        self.one_hot = one_hot
        self.nlabels = nlabels
        self.training_sets: List[Dataset] = []
        self.validation_sets: List[Dataset] = []
        self.test_sets: List[Dataset] = []
        self.ntasks = 0
        self.one_hot = one_hot
        if tasks is not None:
            for task in tasks:
                self.append(task)

    @property
    def feature_dim(self) -> List[int]:
        if not self.training_sets:
            raise ValueError("There are no tasks yet.")
        sample, _ = next(iter(self.training_sets[0]))
        return [s.value for s in sample.shape]

    def append(self, task: Task) -> None:
        """ Append a training set test set pair to the sequence. """
        self.ntasks += 1
        if self.one_hot:
            task = to_one_hot(task, self.nlabels)
        self.training_sets.append(task.train)
        self.validation_sets.append(task.valid)
        self.test_sets.append(task.test)
        self.labels_per_task.append(task.labels)

    def _evaluate(self, model: Model, test: bool = False,
                  logdir: Optional[str] = None) -> Tuple[Array, Array]:
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
        if test:
            evaluation_sets = self.test_sets
        else:
            evaluation_sets = self.validation_sets
        # initialize to -1
        accuracy_matrix = np.zeros((self.ntasks + 1, self.ntasks)) - 1
        average_accuracy = np.zeros((self.ntasks,)) - 1
        accuracy_matrix[0, :] = evaluate(model, evaluation_sets)
        parameters = zip(
            self.training_sets, self.validation_sets, self.labels_per_task)
        eval_sets_until_now = None
        for train_idx, (params, eval_set) in enumerate(zip(parameters,
                                                           evaluation_sets)):
            model.train(*params)
            if train_idx == 0:
                eval_sets_until_now = eval_set
            else:
                eval_sets_until_now = eval_sets_until_now.concatenate(eval_set)
            accuracy_matrix[train_idx + 1, :] = evaluate(model, evaluation_sets)
            average_accuracy[train_idx] = model.evaluate(eval_sets_until_now)
            save_results(logdir=logdir, test=test,
                         average_accuracy=average_accuracy,
                         accuracy_matrix=accuracy_matrix)
        return average_accuracy, accuracy_matrix

    def validate(self, model: Model,
                 logdir: Optional[str] = None) -> Tuple[Array, Array]:
        """ Evaluate the model using validation set. """
        return self._evaluate(model=model, test=False, logdir=logdir)

    def test(self, model: Model,
             logdir: Optional[str] = None) -> Tuple[Array, Array]:
        """ Evaluate the model using test set. """
        return self._evaluate(model=model, test=True, logdir=logdir)
