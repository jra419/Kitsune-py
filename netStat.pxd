import cython
cimport AfterImage as af
cimport numpy as cnp

cnp.import_array()
ctypedef cnp.double_t DTYPE_t


cdef class netStat:
    cdef list Lambdas
    cdef long HostLimit
    cdef long SessionLimit
    cdef long MAC_HostLimit
    cdef long HostSimplexLimit
    cdef int m
    cdef str attack
    cdef af.incStatDB HT_jit
    cdef af.incStatDB HT_MI
    cdef af.incStatDB HT_H
    cdef af.incStatDB HT_Hp

    cdef cnp.ndarray[DTYPE_t, ndim=1] updateGetStats(self, str IPtype, str srcMAC, str dstMAC,
                                                     str srcIP, str srcProtocol, str dstIP,
                                                     str dstProtocol, int datagramSize,
                                                     double timestamp)
    cdef list getNetStatHeaders(self)
