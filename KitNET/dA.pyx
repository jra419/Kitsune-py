import cython
import sys
import numpy as np
cimport numpy as cnp
import json
from numpy.math cimport INFINITY

cnp.import_array()
DTYPE = np.double

cdef extern from "math.h":
    double sqrt(double x) nogil

cdef class dA_params:
    def __init__(self, int n_visible = 5, int n_hidden = 3, double lr=0.001,
                 double corruption_level=0.0, int grace_period = 10000, double hidden_ratio=0):
        self.n_visible = n_visible# num of units in visible (input) layer
        self.n_hidden = n_hidden# num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.grace_period = grace_period
        self.hidden_ratio = hidden_ratio


cdef class dA:
    def __init__(self, dA_params params):
        self.params = params

        if self.params.hidden_ratio != 0:
            self.params.n_hidden = int(np.ceil(self.params.n_visible*self.params.hidden_ratio))

        # for 0-1 normlaization
        # self.norm_max = np.ones((self.params.n_visible,), dtype=DTYPE) * - np.Inf
        # self.norm_min = np.ones((self.params.n_visible,), dtype=DTYPE) * np.Inf
        self.norm_max = np.ones((self.params.n_visible,), dtype=DTYPE) * - INFINITY
        self.norm_min = np.ones((self.params.n_visible,), dtype=DTYPE) * INFINITY
        self.n = 0

        self.rng = np.random.RandomState(1234)

        a = 1. / self.params.n_visible
        self.W = np.array(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.params.n_visible, self.params.n_hidden)), dtype=DTYPE)

        self.hbias = np.zeros(self.params.n_hidden, dtype=DTYPE)  # initialize h bias 0
        self.vbias = np.zeros(self.params.n_visible, dtype=DTYPE)  # initialize v bias 0
        self.W_prime = self.W.T

    cdef cnp.ndarray[DTYPE_t, ndim=1] sigmoid(self, cnp.ndarray[DTYPE_t, ndim=1] x):
        return 1. / (1 + np.exp(-x))

    cdef cnp.ndarray[DTYPE_t, ndim=1] get_corrupted_input(self, input, double corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=input.shape,
                                 n=1,
                                 p=1 - corruption_level) * input

    # Encode
    cdef cnp.ndarray[DTYPE_t, ndim=1] get_hidden_values(self, cnp.ndarray[DTYPE_t, ndim=1] input):
        return self.sigmoid(np.dot(input, self.W) + self.hbias)

    # Decode
    cdef cnp.ndarray[DTYPE_t, ndim=1] get_reconstructed_input(self,
                                                              cnp.ndarray[DTYPE_t, ndim=1] hidden):
        return self.sigmoid(np.dot(hidden, self.W_prime) + self.vbias)

    cpdef double train(self, cnp.ndarray[DTYPE_t, ndim=1] x):
        self.n = self.n + 1
        # update norms
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

        cdef cnp.ndarray[DTYPE_t, ndim=1] tilde_x
        cdef cnp.ndarray[DTYPE_t, ndim=1] y
        cdef cnp.ndarray[DTYPE_t, ndim=1] z
        cdef cnp.ndarray[DTYPE_t, ndim=1] L_h2
        cdef cnp.ndarray[DTYPE_t, ndim=1] L_h1
        cdef cnp.ndarray[DTYPE_t, ndim=1] L_vbias
        cdef cnp.ndarray[DTYPE_t, ndim=1] L_hbias
        cdef cnp.ndarray[DTYPE_t, ndim=2] L_W

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = np.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = np.outer(tilde_x.T, L_h1) + np.outer(L_h2.T, y)

        self.W += self.params.lr * L_W
        self.hbias += self.params.lr * L_hbias
        self.vbias += self.params.lr * L_vbias
        # if np.sqrt(np.mean(L_h2**2, dtype=DTYPE), dtype=DTYPE) > 1:
        if sqrt(np.mean(L_h2**2, dtype=DTYPE)) > 1:
            # print(np.sqrt(np.mean(L_h2**2, dtype=DTYPE), dtype=DTYPE))
            print(sqrt(np.mean(L_h2**2, dtype=DTYPE)))
        # return np.sqrt(np.mean(L_h2**2, dtype=DTYPE), dtype=DTYPE)  # the RMSE reconstruction error during training
        return sqrt(np.mean(L_h2**2, dtype=DTYPE))  # the RMSE reconstruction error during training

    cdef cnp.ndarray[DTYPE_t, ndim=1] reconstruct(self, cnp.ndarray[DTYPE_t, ndim=1] x):
        cdef cnp.ndarray[DTYPE_t, ndim=1] y
        cdef cnp.ndarray[DTYPE_t, ndim=1] z
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        return z

    cpdef double execute(self, cnp.ndarray[DTYPE_t, ndim=1] x):  # returns MSE of the reconstruction of x
        cdef cnp.ndarray[DTYPE_t, ndim=1] z
        cdef double rmse
        if self.n < self.params.grace_period:
            return 0.0
        else:
            # 0-1 normalize
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
            z = self.reconstruct(x)
            # rmse = np.sqrt(((x - z) ** 2).mean(), dtype=DTYPE)  # RMSE
            rmse = sqrt(((x - z) ** 2).mean())  # RMSE
            if rmse > 1:
                print(rmse)
            return rmse

    cdef bint inGrace(self):
        return self.n < self.params.grace_period
