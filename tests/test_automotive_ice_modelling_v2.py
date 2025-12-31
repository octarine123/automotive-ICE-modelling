import numpy as np
import pytest
import math
from src.automotive_ice_modelling_v2 import find_nearest


def test_find_nearest() -> None:
    array = np.array([1, 3, 5, 7, 9])
    test_cases = [
        (2, 1),
        (4, 3),
        (6, 5),
        (8, 7),
        (10, 9),
        (0, 1),
        (5, 5),
        (7, 7),
    ]
    for value, expected in test_cases:
        result = find_nearest(array, value)
        assert result == expected, f"Test failed for input {value}: expected {expected}, got {result}"
    print("All tests passed!")

