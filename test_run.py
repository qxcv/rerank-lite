#!/usr/bin/env python3
"""Unit tests for the reranker. Run these with py.test."""

import numpy as np

from run import unobjectify


def test_unobjectify():
    real_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    # Yup, constructing these arrays is actually really hard
    obj_array = np.ndarray((4, 3), dtype='object')  # yapf: disable
    for idx in range(len(real_data)):
        obj_array[idx] = np.array(real_data[idx], dtype='uint8')
    as_u8 = unobjectify(obj_array)
    assert as_u8.shape == (4, 3)
    assert np.all(as_u8 == np.array(real_data))
