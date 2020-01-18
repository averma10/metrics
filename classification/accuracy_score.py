#!/usr/bin/env python3
import numpy as np


def _nmatches(actual: np.array or list, predicted: np.array or list) -> int:
    """
    Counts the number of matches in the two lists.

    Returns: number of matches in two lists.
    """
    count = 0
    for val1, val2 in zip(actual, predicted):
        if val1 == val2:
            count += 1

    return count


def accuracy_score(actual: np.array or list, predicted: np.array or list, normalize: bool = True) -> float or int:
    """
    Calculate the classification accuracy by number of correct predictions.

    The predicted label must be exactly equal to the actual label.
    Args:
        actual: true label
        predicted: predicted label by classifier model.
        normalize: (default=True)
        If `False` then return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    Returns: Classification Accuracy score

    Raises ValueError: If length of actual is not equal to length of predicted.

    """

    if not len(actual) == len(predicted):
        raise ValueError(f"Must provide arrays of same length. Unable to compare {len(actual)} len array \
                        with {len(predicted)} len array.")

    matches = _nmatches(actual, predicted)
    if normalize:
        return matches / len(actual)
    else:
        return matches
