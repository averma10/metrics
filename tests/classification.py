#!/usr/bin/env python3
import numpy as np
import unittest

from ..classification.accuracy_score import accuracy_score


class TestClassification(unittest.TestCase):
    """
    Test cases to smoke test the classification metrics.

    """

    def setUp(self) -> None:
        """
        This method is called before each test

        """

        self.arr1 = [0, 2, 1, 3]
        self.arr2 = [0, 1, 2, 3]
        self.arr3 = [0, 1, 2]

    def test_accuracy_score(self):
        self.assertEqual(accuracy_score(self.arr1, self.arr2),
                         0.5, "Should be 0.5")

    def test_accuracy_score_normalize_false(self):
        self.assertEqual(accuracy_score(self.arr1, self.arr2, normalize=False),
                         2, "Should be 2")

    def test_accuracy_score_throws_ValueError(self):
        with self.assertRaises(ValueError):
            accuracy_score(self.arr1, self.arr3)


if __name__ == '__main__':
    unittest.main()
