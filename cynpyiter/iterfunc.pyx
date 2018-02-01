#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :iterfunc.pyx
# description            :Example: multithreading the npyiter protocol.
# author                 :Benjamin Winkel
#
# ####################################################################
#  Copyright (C) 2018+ by Benjamin Winkel
#  bwinkel@mpifr.de
#  This file is part of cynpyiter.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ####################################################################

# import python3 compat modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
cimport numpy as np
import cython
cimport cython
from . cimport npyiter
from cython.parallel import parallel
from cpython.ref cimport PyObject
# from cython.operator cimport dereference as deref
cimport openmp
from libc.stdlib cimport abort, malloc, free
from libc.math cimport atan2

np.import_array()


__slots__ = ['threaded_arctan2']


@cython.boundscheck(False)
def threaded_arctan2(np.ndarray A, np.ndarray B):

    cdef:

        # np.ndarray[np.float64_t] ret
        np.ndarray ret
        # PyObject ret

        PyObject *op[3]
        np.uint32_t flags
        np.uint32_t op_flags[3]
        npyiter.PyArray_Descr* op_dtypes[3]

        size_t total_iter_size
        size_t nthreads, th_num

        size_t thr, it_block_size
        np.npy_intp it_start, it_end

        np.npy_intp size

        np.float64_t Ae, Be

        char **errmsg

        int i

        npyiter.NpyIter *c_api_iter, *c_api_iter_thread
        npyiter.IterNextFunc _next
        char **data
        np.npy_intp * strides
        np.npy_intp * size_ptr
        int nop, iop


    # ret = np.zeros_like(A)

    nop = 3
    op[0] = <PyObject*> A
    op[1] = <PyObject*> B
    # op[2] = <PyObject*> C
    op[2] = NULL
    op_flags[0] = npyiter.NPY_ITER_READONLY
    op_flags[1] = npyiter.NPY_ITER_READONLY
    # op_flags[2] = npyiter.NPY_ITER_WRITEONLY
    op_flags[2] = npyiter.NPY_ITER_WRITEONLY | npyiter.NPY_ITER_ALLOCATE

    for i in range(nop):

        if op[i] is NULL:
            op_dtypes[i] = NULL
            continue

        op_dtypes[i] = npyiter.PyArray_DTYPE(<npyiter.PyArrayObject*> op[i])

    # build multi-iter; need a copy later for each thread
    c_api_iter = npyiter.MultiNew(
        # nop:
        nop,
        # op:
        <npyiter.PyArrayObject**> op,
        # flags:
        npyiter.NPY_ITER_EXTERNAL_LOOP |
        npyiter.NPY_ITER_RANGED |
        npyiter.NPY_ITER_BUFFERED,
        # order:
        npyiter.NPY_KEEPORDER,
        # casting:
        npyiter.NPY_NO_CASTING,
        # opflags:
        op_flags,
        # op_dtypes
        <npyiter.PyArray_Descr**> op_dtypes
        )
    if (c_api_iter == NULL):
        raise RuntimeError('Could not instantiate C-API iterator.')

    total_iter_size = npyiter.GetIterSize(c_api_iter)

    # openmp.omp_set_num_threads(2)
    openmp.omp_set_dynamic(1)

    with nogil, parallel():

        nthreads = openmp.omp_get_num_threads()  # works only inside parallel
        th_num = openmp.omp_get_thread_num()  # works only inside parallel

        it_block_size = int(float(total_iter_size) / nthreads + 0.5)

        # we need a copy of the C iterator for each thread, to be able to
        # call the ResetToIterIndexRange function
        with gil:
            c_api_iter_thread = npyiter.Copy(c_api_iter)
            if (c_api_iter_thread == NULL):
                abort()

        size_ptr = npyiter.GetInnerLoopSizePtr(c_api_iter_thread)
        _next = <npyiter.IterNextFunc> npyiter.GetIterNext(
            c_api_iter_thread, errmsg
            )
        data = npyiter.GetDataPtrArray(c_api_iter_thread)
        strides = npyiter.GetInnerStrideArray(c_api_iter_thread)
        size_ptr = npyiter.GetInnerLoopSizePtr(c_api_iter_thread)
        nop = npyiter.GetNOp(c_api_iter_thread)

        it_start = th_num * it_block_size
        it_end = (th_num + 1) * it_block_size

        if th_num == nthreads - 1:
            it_end = total_iter_size

        size = size_ptr[0]

        if not npyiter.ResetToIterIndexRange(
                c_api_iter_thread, it_start, it_end, NULL
                ):
            # error !
            # pass
            abort()

        size = size_ptr[0]

        while size > 0:

            while size > 0:

                Ae = (<np.float64_t*> data[0])[0]
                Be = (<np.float64_t*> data[1])[0]

                # don't ever use a prange in here!!!
                (<np.float64_t*> data[2])[0] = atan2(Ae, Be)

                for iop in range(nop):
                    data[iop] += strides[iop]

                size -= 1

            _next(c_api_iter_thread)
            size = size_ptr[0]

        with gil:
            if (npyiter.Deallocate(c_api_iter_thread) != npyiter.NPY_SUCCEED):
                # __Pyx_DECREF(ret)
                abort()

    ret = <object> (npyiter.GetOperandArray(c_api_iter)[2])

    if (npyiter.Deallocate(c_api_iter) != npyiter.NPY_SUCCEED):
        # cython.Py_DECREF(ret)
        # abort()
        return None

    return ret
