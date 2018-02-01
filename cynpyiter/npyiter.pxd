#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :npyiter.pxd
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

cimport numpy as np
from cpython.ref cimport PyObject


ctypedef void NpyIter
ctypedef void PyArrayObject
ctypedef void PyArray_Descr
ctypedef int (*IterNextFunc)(NpyIter * iter) nogil
ctypedef void (*GetMultiIndexFunc)(
    NpyIter * iter, np.npy_intp *outcoords
    ) nogil


cdef extern from "numpy/arrayobject.h":

    # Per-operand flags that may be passed to the iterator constructors

    # The operand will be read from and written to
    cpdef int NPY_ITER_READWRITE
    # The operand will only be read from
    cpdef int NPY_ITER_READONLY
    # The operand will only be written to
    cpdef int NPY_ITER_WRITEONLY
    # The operand's data must be in native byte order
    cpdef int NPY_ITER_NBO
    # The operand's data must be aligned
    cpdef int NPY_ITER_ALIGNED
    # The operand's data must be contiguous (within the inner loop)
    cpdef int NPY_ITER_CONTIG
    # The operand may be copied to satisfy requirements
    cpdef int NPY_ITER_COPY
    # The operand may be copied with WRITEBACKIFCOPY to satisfy
    # requirements
    cpdef int NPY_ITER_UPDATEIFCOPY
    # Allocate the operand if it is NULL
    cpdef int NPY_ITER_ALLOCATE
    # If an operand is allocated, don't use any subtype
    cpdef int NPY_ITER_NO_SUBTYPE
    # This is a virtual array slot, operand is NULL but temporary data is
    # there
    cpdef int NPY_ITER_VIRTUAL
    # Require that the dimension match the iterator dimensions exactly
    cpdef int NPY_ITER_NO_BROADCAST
    # A mask is being used on this array, affects buffer -> array copy
    cpdef int NPY_ITER_WRITEMASKED
    # This array is the mask for all WRITEMASKED operands
    cpdef int NPY_ITER_ARRAYMASK
    # Assume iterator order data access for COPY_IF_OVERLAP
    cpdef int NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE

    # For specifying allowed casting in operations which support it
    cpdef enum NPY_CASTING:
        # Only allow identical types
        NPY_NO_CASTING
        # Allow identical and byte swapped types
        NPY_EQUIV_CASTING
        # Only allow safe casts
        NPY_SAFE_CASTING
        # Allow safe casts or casts within the same kind
        NPY_SAME_KIND_CASTING
        # Allow any casts
        NPY_UNSAFE_CASTING

    # For specifying array memory layout or iteration order
    cpdef enum NPY_ORDER:
        # Fortran order if inputs are all Fortran, C otherwise
        NPY_ANYORDER
        # C order
        NPY_CORDER
        # Fortran order
        NPY_FORTRANORDER
        # An order as close to the inputs as possible
        NPY_KEEPORDER

    # Global flags that may be passed to the iterator constructors:

    # Track an index representing C order
    cpdef int NPY_ITER_C_INDEX
    # Track an index representing Fortran order
    cpdef int NPY_ITER_F_INDEX
    # Track a multi-index
    cpdef int NPY_ITER_MULTI_INDEX
    # User code external to the iterator does the 1-dimensional
    # innermost loop
    cpdef int NPY_ITER_EXTERNAL_LOOP
    # Convert all the operands to a common data type
    cpdef int NPY_ITER_COMMON_DTYPE
    # Operands may hold references, requiring API access during iteration
    cpdef int NPY_ITER_REFS_OK
    # Zero-sized operands should be permitted, iteration checks
    # IterSize for 0
    cpdef int NPY_ITER_ZEROSIZE_OK
    # Permits reductions (size-0 stride with dimension size > 1)
    cpdef int NPY_ITER_REDUCE_OK
    # Enables sub-range iteration
    cpdef int NPY_ITER_RANGED
    # Enables buffering
    cpdef int NPY_ITER_BUFFERED
    # When buffering is enabled, grows the inner loop if possible
    cpdef int NPY_ITER_GROWINNER
    # Delay allocation of buffers until first Reset* call
    cpdef int NPY_ITER_DELAY_BUFALLOC
    # When NPY_KEEPORDER is specified, disable reversing
    # negative-stride axes
    cpdef int NPY_ITER_DONT_NEGATE_STRIDES
    # If output operands overlap with other operands (based on heuristics
    # that has false positives but no false negatives), make temporary
    # copies to eliminate overlap.
    cpdef int NPY_ITER_COPY_IF_OVERLAP

    # Used for Converter Functions "O&" code in ParseTuple
    cpdef int NPY_SUCCEED
    cpdef int NPY_FAIL

    IterNextFunc GetIterNext "NpyIter_GetIterNext" (
        NpyIter *iter, char **
        ) nogil
    char** GetDataPtrArray "NpyIter_GetDataPtrArray" (NpyIter* iter) nogil
    np.npy_intp * GetInnerStrideArray "NpyIter_GetInnerStrideArray" (
        NpyIter* iter
        ) nogil
    np.npy_intp * GetInnerLoopSizePtr "NpyIter_GetInnerLoopSizePtr" (
        NpyIter* iter
        ) nogil
    int GetNDim "NpyIter_GetNDim" (NpyIter* iter) nogil
    int GetNOp "NpyIter_GetNOp" (NpyIter* iter) nogil

    int ResetToIterIndexRange "NpyIter_ResetToIterIndexRange" (
        NpyIter * iter,
        np.npy_intp start, np.npy_intp end,
        char ** errmsg
        ) nogil

    np.npy_intp GetIterSize "NpyIter_GetIterSize" (NpyIter* iter)
    NpyIter* Copy "NpyIter_Copy" (NpyIter* iter)
    np.npy_bool IsBuffered "NpyIter_IsBuffered" (NpyIter * iter)
    np.npy_intp GetBufferSize "NpyIter_GetBufferSize" (NpyIter * iter)
    int Deallocate "NpyIter_Deallocate" (NpyIter* iter)

    NpyIter* MultiNew "NpyIter_MultiNew" (
        np.npy_intp nop,
        PyArrayObject** op,
        np.uint32_t flags,
        NPY_ORDER order,  # np.NPY_ORDER order,
        NPY_CASTING casting,  # np.NPY_CASTING casting,
        np.uint32_t* op_flags,
        PyArray_Descr** op_dtypes
        )

    NpyIter* AdvancedNew "NpyIter_AdvancedNew" (
        np.npy_intp nop,
        PyArrayObject** op,
        np.uint32_t flags,
        NPY_ORDER order,
        NPY_CASTING casting,
        np.uint32_t* op_flags,
        PyArray_Descr** op_dtypes,
        int oa_ndim,
        int** op_axes,
        np.npy_intp* itershape,
        np.npy_intp buffersize
        )

    PyObject** GetOperandArray "NpyIter_GetOperandArray" (NpyIter* iter)
    PyArray_Descr *PyArray_DTYPE(PyArrayObject* arr)
    int PyArray_TYPE(PyArrayObject* arr)
