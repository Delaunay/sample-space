#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for unittests and functional tests."""

import numpy
import pytest


@pytest.fixture(scope="function")
def seed():
    """Return a fixed ``numpy.random.RandomState`` and global seed."""
    seed = 5
    rng = numpy.random.RandomState(seed)
    numpy.random.seed(seed)
    return rng
