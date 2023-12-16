import cython
cimport netStat as nS
cimport numpy as cnp

cnp.import_array()
ctypedef cnp.double_t DTYPE_t


cdef class FE:
    cdef str path
    cdef double limit
    cdef int fm_grace
    cdef int ad_grace
    cdef bint train_skip
    cdef int offset
    cdef str train_stats
    cdef int curPacketIndx
    cdef nS.netStat nstat
    cdef df_csv

    cdef void __check_csv__(self)
    cdef void parse_pcap(self)
    cdef list get_next_vector(self, int flag)
