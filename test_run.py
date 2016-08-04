#!/usr/bin/env python3
"""Unit tests for the reranker. Run these with py.test."""

from os import path

import pytest

import numpy as np

import run

MPII_DATA_ROOT = 'data/'
MPII_JSON_PATH = path.join(MPII_DATA_ROOT, 'mpii_human_pose_v1_u12_1.json')


@pytest.mark.skipif(not path.exists(MPII_JSON_PATH),
                    reason='need MPII data to test dataset loading')
def test_dataset():
    dataset = run.PoseDataset(MPII_DATA_ROOT)
    annots = dataset[...]
    assert isinstance(annots, np.ndarray)
    train_inds = dataset.mpii_train_indices
    test_inds = dataset.mpii_test_indices
    assert len(annots) == len(train_inds) + len(test_inds)

    some_idx = dataset[0]
    assert isinstance(some_idx, run.DatumAnnotation)
