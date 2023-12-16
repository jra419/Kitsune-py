import numpy as np
cimport numpy as cnp
from scipy.cluster.hierarchy import linkage, fcluster, to_tree

# A helper class for KitNET which performs a correlation-based incremental clustering of the dimensions in X
# n: the number of dimensions in the dataset
# For more information and citation, please see our NDSS'18 paper:
# Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

cnp.import_array()
DTYPE = np.double


cdef class corClust:
    def __init__(self, int n):
        # parameter:
        self.n = n
        # variables
        self.c = np.zeros(n, dtype=DTYPE)  # linear num of features
        self.c_r = np.zeros(n, dtype=DTYPE)  # linear sum of feature residules
        self.c_rs = np.zeros(n, dtype=DTYPE)  # linear sum of feature residules
        self.C = np.zeros((n, n), dtype=DTYPE)  # partial correlation matrix
        self.N = 0  # number of updates performed

    # x: a numpy vector of length n
    cdef void update(self, cnp.ndarray[DTYPE_t, ndim=1] x):
        cdef cnp.ndarray[DTYPE_t, ndim=1] c_rt
        self.N += 1
        self.c += x
        c_rt = x - self.c/self.N
        self.c_r += c_rt
        self.c_rs += c_rt**2
        self.C += np.outer(c_rt, c_rt)

    # creates the current correlation distance matrix between the features
    cdef corrDist(self):
        c_rs_sqrt = np.sqrt(self.c_rs, dtype=DTYPE)
        C_rs_sqrt = np.outer(c_rs_sqrt, c_rs_sqrt)
        # this protects against dive by zero erros (occurs when a feature is a constant)
        C_rs_sqrt[C_rs_sqrt == 0] = 1e-100
        D = 1-self.C/C_rs_sqrt  # the correlation distance matrix
        # small negatives may appear due to the incremental fashion in which we update the mean.
        # Therefore, we 'fix' them
        D[D < 0] = 0
        return D

    # clusters the features together, having no more than maxClust features per cluster
    cdef list cluster(self, int maxClust):
        D = self.corrDist()
        Z = linkage(D[np.triu_indices(self.n, 1)])  # create a linkage matrix based on the distance matrix
        if maxClust < 1:
            maxClust = 1
        if maxClust > self.n:
            maxClust = self.n
        cdef list map
        map = self.__breakClust__(to_tree(Z), maxClust)
        return map


    # a recursive helper function which breaks down the dendrogram branches until all
    # clusters have no more than maxClust elements
    cdef list __breakClust__(self, dendro, int maxClust):
        if dendro.count <= maxClust:  # base case: we found a minimal cluster, so mark it
            return [dendro.pre_order()]  # return the origional ids of the features in this cluster
        return self.__breakClust__(dendro.get_left(), maxClust) + self.__breakClust__(
            dendro.get_right(), maxClust)
