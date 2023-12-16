import cython
cimport numpy as cnp

cnp.import_array()
ctypedef cnp.double_t DTYPE_t

cdef class corClust:
    cdef int n
    cdef c
    cdef c_r
    cdef c_rs
    cdef C
    cdef int N

    cdef void update(self, cnp.ndarray[DTYPE_t, ndim=1] x)
    cdef corrDist(self)
    cdef list cluster(self, int maxClust)
    cdef list __breakClust__(self, dendro, int maxClust)
