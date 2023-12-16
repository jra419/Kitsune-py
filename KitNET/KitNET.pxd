# from corClust cimport corClust
cimport KitNET.corClust as CC
cimport KitNET.dA as AE
# from dA cimport dA
cimport numpy as cnp

cnp.import_array()
ctypedef cnp.double_t DTYPE_t

# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper:
# Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document


cdef class KitNET:
    cdef int FM_grace_period
    cdef int AD_grace_period
    cdef int m
    cdef double lr
    cdef double hr
    cdef int n
    cdef int n_trained
    cdef int n_executed
    cdef list ensemble_layer
    cdef AE.dA output_layer
    cdef str attack
    cdef CC.corClust FM
    cdef list v

    cdef double process(self, cnp.ndarray[DTYPE_t, ndim=1] x)
    cdef double train(self, cnp.ndarray[DTYPE_t, ndim=1] x)
    cdef double execute(self, cnp.ndarray[DTYPE_t, ndim=1] x)
    cdef void createAD(self)
