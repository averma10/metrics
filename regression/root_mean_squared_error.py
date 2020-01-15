#!/usr/bin/env python3
import numpy as np
from .mean_squared_error import mean_squared_error


def root_mean_squared_error(actual: np.array, predicted: np.array) -> float:
    """
    calculate the root mean squared error.
    Args:
        actual: true label
        predicted: predicted label

    Returns: root mean squared error

    """

    return np.sqrt(mean_squared_error(actual, predicted))
