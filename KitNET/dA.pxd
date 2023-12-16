import cython
cimport numpy as cnp

cnp.import_array()
ctypedef cnp.double_t DTYPE_t


cdef class dA_params:
    cdef public int n_visible
    cdef public int n_hidden
    cdef public double lr
    cdef public double corruption_level
    cdef public int grace_period
    cdef public double hidden_ratio


cdef class dA:
    cdef public dA_params params
    cdef public norm_max
    cdef public norm_min
    cdef public int n
    cdef public rng
    cdef public W
    cdef public hbias
    cdef public vbias
    cdef public W_prime

    cdef cnp.ndarray[DTYPE_t, ndim=1] sigmoid(self, cnp.ndarray[DTYPE_t, ndim=1] x)
    cdef cnp.ndarray[DTYPE_t, ndim=1] get_corrupted_input(self, input, double corruption_level)
    cdef cnp.ndarray[DTYPE_t, ndim=1] get_hidden_values(self, cnp.ndarray[DTYPE_t, ndim=1] input)
    cdef cnp.ndarray[DTYPE_t, ndim=1] get_reconstructed_input(self,
                                                              cnp.ndarray[DTYPE_t, ndim=1] hidden)
    cpdef double train(self, cnp.ndarray[DTYPE_t, ndim=1] x)
    cdef cnp.ndarray[DTYPE_t, ndim=1] reconstruct(self, cnp.ndarray[DTYPE_t, ndim=1] x)
    cpdef double execute(self, cnp.ndarray[DTYPE_t, ndim=1] x)
    cdef bint inGrace(self)
