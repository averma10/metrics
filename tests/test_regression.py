#!/usr/bin/env python3
import unittest
import numpy as np

from ..regression.mean_absolute_error import mean_absolute_error
from ..regression.mean_squared_error import mean_squared_error


class TestRegression(unittest.TestCase):
    """
    Test case to smoke test the regression metrics
    """
    def test_mean_absolute_error(self):
        self.assertEqual(mean_absolute_error(np.array([2, 4]), np.array([3, 1])),
                         2.0, "Should be 2.0")

    def test_mean_squared_error(self):
        self.assertEqual(mean_squared_error(np.array([2, 4]), np.array([3,1])),
                         5.0, "Should be 5.0")


if __name__ == '__main__':
    unittest.main()
