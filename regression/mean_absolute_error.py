#!/usr/bin/env python3
import numpy as np


def absolute_error(actual: np.array, predicted: np.array) -> np.array:
    """
    calculate absolute error
    Args:
        actual: true label
        predicted: predicted label

    Returns: array of absolute error

    """

    return np.abs(actual - predicted)


def mean_absolute_error(actual: np.array, predicted: np.array) -> float:
    """
    calculate mean absolute error
    Args:
        actual: true label
        predicted: predicted label

    Returns: mean absolute error

    """

    return np.mean(absolute_error(actual, predicted))
