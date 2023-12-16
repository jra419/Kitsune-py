import cython

cdef class incStat:
    cdef str ID
    cdef double CF1
    cdef double CF2
    cdef double w
    cdef int isTypeDiff
    cdef double Lambda
    cdef double lastTimestamp
    cdef double cur_mean
    cdef double cur_var
    cdef double cur_std
    cdef list covs

    cdef void insert(self, double v, double t=*)
    cdef double processDecay(self, double timestamp)
    cdef double weight(self)
    cdef double mean(self)
    cdef double var(self)
    cdef double std(self)
    cdef list cov(self, str ID2)
    cdef list pcc(self, str ID2)
    cdef list cov_pcc(self, str ID2)
    cdef double radius(self, list other_incStats)
    cdef double magnitude(self, list other_incStats)
    cdef list allstats_1D(self)
    cdef list allstats_2D(self, str ID2)
    cdef list getHeaders_1D(self, bint suffix=*)
    cdef list getHeaders_2D(self, str ID2, bint suffix=*)

# Like incStat, but maintains stats between two streams
cdef class incStat_cov:
    cdef incStat incS1
    cdef incStat incS2
    cdef double CF3
    cdef double w3
    cdef double lastTimestamp_cf3
    # cdef list incStats
    cdef list lastRes

    cdef void update_cov(self, str ID, double v, double t)
    cdef double processDecay(self, double t, int micro_inc_indx)
    cdef double cov(self)
    cdef double pcc(self)
    cdef list get_stats1(self)
    cdef list get_stats2(self)
    cdef list get_stats3(self)
    cdef list get_stats4(self)
    cdef list getHeaders(self, int ver, bint suffix=*)

cdef class incStatDB:
    cdef double limit
    cdef double df_lambda
    cdef dict HT

    cdef dict get_dict(self)
    cdef double get_lambda(self, double Lambda)
    cdef incStat register(self, str ID, double Lambda=*, double init_time=*,
                          bint isTypeDiff=*)
    cdef incStat_cov register_cov(self, str ID1, str ID2, double Lambda=*,
                                  double init_time=*, bint isTypeDiff=*)
    cdef incStat update(self, str ID, double t, double v, double Lambda=*,
                        bint isTypeDiff=*)
    cdef list get_1D_Stats(self, str ID, double Lambda=*)
    cdef list get_2D_Stats(self, str ID1, str ID2, double Lambda=*)
    cdef tuple get_all_2D_Stats(self, str ID, double Lambda=*)
    cdef list get_nD_Stats(self, list IDs, double Lambda=*)
    cdef list update_get_1D_Stats(self, str ID, double t, double v,
                                  double Lambda=*, bint isTypeDiff=*)
    cdef list update_get_2D_Stats(self, str ID1, str ID2, double t1, double v1,
                                  double Lambda=*, int level=*)
    cdef list update_get_1D2D_Stats(self, str ID1, str ID2, double t1, double v1, double Lambda=*)
    cdef list getHeaders_1D(self, double Lambda=*, str ID=*)
    cdef list getHeaders_2D(self, double Lambda=*, list IDs=*, int ver=*)
    cdef list getHeaders_1D2D(self, double Lambda=*, list IDs=*, int ver=*)
    cdef list getHeaders_nD(self, double Lambda=*, list IDs=*)
    cdef double clean_out_old_records(self, double cutoffWeight)
