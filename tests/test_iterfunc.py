#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
# from astropy.utils.misc import NumpyRNGContext

from cynpyiter import threaded_arctan2


def test_threaded_arctan2():

    size = (2, 4, 8, 2 ** 16)

    arr1 = np.random.uniform(size=size)
    arr2 = np.random.uniform(size=size)

    res1 = threaded_arctan2(arr1, arr2)
    res2 = np.arctan2(arr1, arr2)

    assert_allclose(res1, res2)
