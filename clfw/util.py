import os.path
from typing import Optional

import numpy as np


def _build_path(logdir: str, test: bool) -> str:
    if test:
        name = 'test_result.npz'
    else:
        name = 'valid_result.npz'
    return os.path.join(logdir, name)


def save_results(logdir: Optional[str], test: bool,
                 average_accuracy: np.ndarray,
                 accuracy_matrix: np.ndarray) -> None:
    if logdir is None:
        return
    path = _build_path(logdir=logdir, test=test)
    np.savez(path,
             average_accuracy=average_accuracy,
             accuracy_matrix=accuracy_matrix)


def print_results(logdir: str, test: bool = False) -> None:
    path = _build_path(logdir=logdir, test=test)
    results = np.load(path)
    print('------------------------------------------------------------------')
    print(path)
    print('average_accuracy')
    print(results['average_accuracy'])
    print('accuracy_matrix')
    print(results['accuracy_matrix'])
