#!/usr/bin/env python3
import numpy as np


def squared_error(actual: np.array, predicted: np.array) -> np.array:
    """
    calculate the squared error.
    Args:
        actual: actual labels
        predicted: predicted labels

    Returns: Squared Error

    """

    return (actual - predicted)**2


def sum_squared_error(actual: np.array, predicted: np.array) -> int:
    """
    calculate the sum of squared error.
    Args:
        actual: actual labels
        predicted: predicted labels

    Returns: sum of squared error

    """

    return np.sum(squared_error(actual, predicted))


def mean_squared_error(actual: np.array, predicted: np.array) -> float:
    """
    calculate mean squared error.
    Args:
        actual: actual labels
        predicted: predicted labels

    Returns:
        Mean Squared Error from provided params

    """
    return np.mean(squared_error(actual, predicted))