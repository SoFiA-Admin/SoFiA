#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# distutils: language = c++
# distutils: sources = CNHI.h

# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

cimport numpy as np
import numpy as np

cdef extern from "CNHI.h":

    cdef void CNHI_find_sources(float * data, int * mask, int verbose, int metric[9], float pReq, int tr_min, int tr_max, int flag_val, bint medianTest, float qReq)

def find_sources(data, mask, pReq = 0.05, minScale = 5, maxScale = -1, verbose = 1, medianTest = 1, qReq = 3.8, flag_val = 1):

    nan_mask = np.isnan(data)
    found_nan = nan_mask.sum()
    if found_nan:
        data = np.nan_to_num(data)
        mask = find_sources(data.astype(np.single, copy = False), mask.astype(np.intc, copy = False), pReq, minScale, maxScale, verbose, flag_val, medianTest, qReq)
        data[nan_mask] = np.nan
    else:
        mask = _find_sources(data.astype(np.single, copy = False), mask.astype(np.intc, copy = False), pReq, minScale, maxScale, verbose, flag_val, medianTest, qReq)

    return mask

cdef _find_sources(np.ndarray[dtype = float, ndim = 3] data, np.ndarray[dtype = int, ndim = 3] mask, float pReq, int minScale, int maxScale, int verbose, int flag_val, int medianTest, float qReq):

    # define variables
    cdef bint c_medianTest
    cdef int metric[9]

    # set the boolean flag, c_medianTest, that sets whether detections must have a median greater than the median of the remaining data
    c_medianTest = 0
    if (medianTest > 0): c_medianTest = 1

    # set the metric used to parse the datacube
    metric[0] = data.shape[2]
    metric[1] = data.shape[1]
    metric[2] = data.shape[0]
    metric[3] = 0
    metric[4] = 0
    metric[5] = 0
    metric[6] = data.shape[2]
    metric[7] = data.shape[1]
    metric[8] = data.shape[0]

    # call the function
    CNHI_find_sources(<float *> data.data, <int *> mask.data, verbose, metric, <float> pReq, minScale, maxScale, <int> flag_val, c_medianTest, <float> qReq)

    return mask
