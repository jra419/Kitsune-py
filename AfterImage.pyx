import math
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
from libc.math cimport isnan, pow, sqrt


cdef class incStat:
    def __init__(self, double Lambda, str ID, double init_time=0, int isTypeDiff=False):  # timestamp is creation time
        self.ID = ID
        self.CF1 = 0  # linear sum
        self.CF2 = 0  # sum of squares
        self.w = 1e-20  # weight
        self.isTypeDiff = isTypeDiff
        self.Lambda = Lambda  # Decay Factor
        self.lastTimestamp = init_time
        self.cur_mean = np.nan
        self.cur_var = np.nan
        self.cur_std = np.nan
        self.covs = []  # a list of incStat_covs (references) with relate to this incStat

    cdef void insert(self, double v, double t=0):  # v is a scalar, t is v's arrival the timestamp
        if self.isTypeDiff:
            if t - self.lastTimestamp > 0:
                v = t - self.lastTimestamp
            else:
                v = 0
        self.processDecay(t)

        # update with v
        self.CF1 += v
        self.CF2 += pow(v, 2)
        self.w += 1
        self.cur_mean = np.nan  # force recalculation if called
        self.cur_var = np.nan
        self.cur_std = np.nan

        # update covs (if any)
        cdef incStat_cov cov
        for c in self.covs:
            cov = c
            cov.update_cov(self.ID, v, t)

    cdef double processDecay(self, double timestamp):
        factor = 1
        # check for decay
        timeDiff = timestamp - self.lastTimestamp
        if timeDiff > 0:
            factor = pow(2, (-self.Lambda * timeDiff))
            self.CF1 = self.CF1 * factor
            self.CF2 = self.CF2 * factor
            self.w = self.w * factor
            self.lastTimestamp = timestamp
        return factor

    cdef double weight(self):
        return self.w

    cdef double mean(self):
        if isnan(self.cur_mean):  # calculate it only once when necessary
            if self.w != 0:
                self.cur_mean = self.CF1 / self.w
            else:
                self.cur_mean = 0
        return self.cur_mean

    cdef double var(self):
        if isnan(self.cur_var):  # calculate it only once when necessary
            w = self.w - pow(self.mean(), 2)
            if w != 0:
                self.cur_var = abs(self.CF2 / w)
            else:
                self.cur_var = 0
        return self.cur_var

    cdef double std(self):
        if isnan(self.cur_std):  # calculate it only once when necessary
            self.cur_std = sqrt(self.var())
        return self.cur_std

    cdef list cov(self, str ID2):
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.cov()
        return [np.nan]

    cdef list pcc(self, str ID2):
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.pcc()
        return [np.nan]

    cdef list cov_pcc(self, str ID2):
        cdef incStat_cov cov
        for c in self.covs:
            cov = c
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.get_stats1()
        return [np.nan]*2

    cdef double radius(self, list other_incStats):  # the radius of a set of incStats
        cdef double A
        A = self.var()**2
        cdef incStat incSc
        for incS in other_incStats:
            incSc = incS
            A += incSc.var()**2
        return sqrt(A)

    cdef double magnitude(self, list other_incStats):  # the magnitude of a set of incStats
        cdef double A
        A = pow(self.mean(), 2)
        cdef incStat incSc
        for incS in other_incStats:
            incSc = incS
            A += pow(incSc.mean(), 2)
        return sqrt(A)

    # Calculates and pulls all stats on this stream
    cdef list allstats_1D(self):
        if self.w != 0:
            self.cur_mean = self.CF1 / self.w
        else:
            self.cur_mean = 0
        if (self.w - pow(self.cur_mean, 2)) != 0:
            self.cur_var = abs(self.CF2 / self.w - pow(self.cur_mean, 2))
        else:
            self.cur_var = 0
        return [self.w, self.cur_mean, self.cur_var]

    # Calculates and pulls all stats on this stream, and stats shared with the indicated stream
    cdef list allstats_2D(self, str ID2):
        stats1D = self.allstats_1D()
        # Find cov component
        stats2D = [np.nan] * 4
        cdef incStat_cov cov
        for c in self.covs:
            cov = c
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                stats2D = cov.get_stats2()
                break
        return stats1D + stats2D

    cdef list getHeaders_1D(self, bint suffix=True):
        if self.ID is None:
            s0 = ""
        else:
            s0 = "_0"
        if suffix:
            s0 = "_"+self.ID
        headers = ["weight"+s0, "mean"+s0, "std"+s0]
        return headers

    cdef list getHeaders_2D(self, str ID2, bint suffix=True):
        hdrs1D = self.getHeaders_1D(suffix)
        if self.ID is None:
            s0 = ""
            s1 = ""
        else:
            s0 = "_0"
            s1 = "_1"
        if suffix:
            s0 = "_"+self.ID
            s1 = "_" + ID2
        hdrs2D = ["radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1,
                  "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        return hdrs1D + hdrs2D


# Like incStat, but maintains stats between two streams
cdef class incStat_cov:
    def __init__(self, incStat incS1, incStat incS2, double init_time=0):
        # store references tot he streams' incStats
        # self.incStats = [incS1, incS2]
        self.incS1 = incS1
        self.incS2 = incS2
        self.lastRes = [0, 0]
        # init extrapolators
        # self.EXs = [extrapolator(),extrapolator()]

        # init sum product residuals
        self.CF3 = 0  # sum of residule products (A-uA)(B-uB)
        self.w3 = 1e-20
        self.lastTimestamp_cf3 = init_time

    # Other_incS_decay is the decay factor of the other incstat
    # ID: the stream ID which produced (v,t)
    cdef void update_cov(self, str ID, double v, double t):
        # it is assumes that incStat "ID" has ALREADY been updated with (t,v)
        # [this si performed automatically in method incStat.insert()]

        # find incStat
        cdef int inc
        # if ID == self.incStats[0].ID:
        if ID == self.incS1.ID:
            inc = 0
        elif ID == self.incS2.ID:
        # elif ID == self.incStats[1].ID:
            inc = 1
        else:
            print("update_cov ID error")
            return  # error

        # Decay other incStat
        # self.incStats[not inc].processDecay(t)

        if inc == 0:
            self.incS2.processDecay(t)
        else:
            self.incS1.processDecay(t)

        # Decay residules
        self.processDecay(t, inc)

        # Update extrapolator for current stream
        # self.EXs[inc].insert(t,v)

        # Extrapolate other stream
        # v_other = self.EXs[not(inc)].predict(t)

        # Compute and update residule
        cdef double res
        cdef double resid
        # res = (v - self.incStats[inc].mean())
        # resid = (v - self.incStats[inc].mean()) * self.lastRes[not inc]
        # self.CF3 += resid
        # self.w3 += 1
        # self.lastRes[inc] = res
        if inc == 0:
            res = (v - self.incS1.mean())
            resid = (v - self.incS1.mean()) * self.lastRes[1]
            self.CF3 += resid
            self.w3 += 1
            self.lastRes[0] = res
        else:
            res = (v - self.incS2.mean())
            resid = (v - self.incS2.mean()) * self.lastRes[0]
            self.CF3 += resid
            self.w3 += 1
            self.lastRes[1] = res

    cdef double processDecay(self, double t, int micro_inc_indx):
        cdef double factor
        factor = 1
        # check for decay cf3
        cdef double timeDiffs_cf3
        timeDiffs_cf3 = t - self.lastTimestamp_cf3
        if timeDiffs_cf3 > 0:
            if micro_inc_indx == 0:
                factor = pow(2, (-(self.incS1.Lambda) * timeDiffs_cf3))
            else:
                factor = pow(2, (-(self.incS2.Lambda) * timeDiffs_cf3))
            self.CF3 *= factor
            self.w3 *= factor
            self.lastTimestamp_cf3 = t
            self.lastRes[micro_inc_indx] *= factor
        return factor

    # Todo: add W3 for cf3

    # Covariance approximation
    cdef double cov(self):
        if self.w3 != 0:
            return self.CF3 / self.w3
        else:
            return 0

    # Pearson corl. coef
    cdef double pcc(self):
        cdef double ss
        # ss = self.incStats[0].std() * self.incStats[1].std()
        ss = self.incS1.std() * self.incS2.std()
        if ss != 0:
            return self.cov() / ss
        else:
            return 0

    # calculates and pulls all correlative stats
    cdef list get_stats1(self):
        return [self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND 2D stats from both streams (incStat)
    cdef list get_stats2(self):
        # return [self.incStats[0].radius([self.incStats[1]]),
        #         self.incStats[0].magnitude([self.incStats[1]]), self.cov(), self.pcc()]
        return [self.incS1.radius([self.incS2]),
                self.incS1.magnitude([self.incS2]), self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND 2D stats
    # AND the regular stats from both streams (incStat)
    cdef list get_stats3(self):
        return [self.incS1.w, self.incS1.mean(), self.incS1.std(),
                self.incS2.w, self.incS2.mean(), self.incS2.std(),
                self.cov(), self.pcc()]

    # calculates and pulls all correlative stats
    # AND the regular stats from both incStats AND 2D stats
    cdef list get_stats4(self):
        return [self.incS1.w, self.incS1.mean(), self.incS1.std(),
                self.incS2.w, self.incS2.mean(), self.incS2.std(),
                self.incS1.radius([self.incS2]),
                self.incS1.magnitude([self.incS2]), self.cov(), self.pcc()]

    cdef list getHeaders(self, int ver, bint suffix=True):  # ver = {1,2,3,4}
        headers = []
        s0 = "0"
        s1 = "1"
        if suffix:
            # s0 = self.incStats[0].ID
            # s1 = self.incStats[1].ID
            s0 = self.incS1.ID
            s1 = self.incS2.ID

        if ver == 1:
            headers = ["covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 2:
            headers = ["radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1,
                       "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 3:
            headers = ["weight_" + s0, "mean_" + s0, "std_" + s0, "weight_" + s1, "mean_" + s1,
                       "std_" + s1, "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 4:
            headers = ["weight_" + s0, "mean_" + s0, "std_" + s0, "covariance_" + s0 + "_" + s1,
                       "pcc_" + s0 + "_" + s1]
        if ver == 5:
            headers = ["weight_" + s0, "mean_" + s0, "std_" + s0, "weight_" + s1, "mean_" + s1,
                       "std_" + s1, "radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1,
                       "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        return headers


cdef class incStatDB:
    # default_lambda: use this as the lambda for all streams.
    # If not specified, then you must supply a Lambda with every query.
    def __init__(self, double limit=np.inf, double default_lambda=np.nan):
        self.HT = dict()
        self.limit = limit
        self.df_lambda = default_lambda

    cdef dict get_dict(self):
        return self.HT

    cdef double get_lambda(self, double Lambda):
        if not isnan(self.df_lambda):
            Lambda = self.df_lambda
        return Lambda

    # Registers a new stream. init_time: init lastTimestamp of the incStat
    cdef incStat register(self, str ID, double Lambda=1, double init_time=0,
                          bint isTypeDiff=False):
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Retrieve incStat
        cdef str key
        key = ID+"_"+str(Lambda)
        cdef incStat incS
        incS = self.HT.get(key)
        if incS is None:  # does not already exist
            if len(self.HT) + 1 > self.limit:
                raise LookupError(
                    'Adding Entry:\n' + key + '\nwould exceed incStatHT 1D limit of ' + str(
                        self.limit) + '.\nObservation Rejected.')
            incS = incStat(Lambda, ID, init_time, isTypeDiff)
            self.HT[key] = incS  # add new entry
        return incS

    # Registers covariance tracking for two streams, registers missing streams
    cdef incStat_cov register_cov(self, str ID1, str ID2, double Lambda=1,
                                  double init_time=0, bint isTypeDiff=False):
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Lookup both streams
        cdef incStat incS1
        cdef incStat incS2
        incS1 = self.register(ID1, Lambda, init_time, isTypeDiff)
        incS2 = self.register(ID2, Lambda, init_time, isTypeDiff)

        # Check for pre-exiting link
        # for cov in incS1.covs:
        #     if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
        #         return cov  # there is a pre-exiting link
        cdef incStat_cov cov
        if len(incS1.covs) < len(incS2.covs):
            for c in incS1.covs:
                cov = c
                # if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                if cov.incS1.ID == ID2 or cov.incS2.ID == ID2:
                    return cov  # there is a pre-exiting link
        else:
            for c in incS2.covs:
                cov = c
                # if cov.incStats[0].ID == ID1 or cov.incStats[1].ID == ID1:
                if cov.incS1.ID == ID1 or cov.incS2.ID == ID1:
                    return cov  # there is a pre-exiting link

        # Link incStats
        cdef incStat_cov inc_cov
        inc_cov = incStat_cov(incS1, incS2, init_time)
        incS1.covs.append(inc_cov)
        incS2.covs.append(inc_cov)
        return inc_cov

    # updates/registers stream
    cdef incStat update(self, str ID, double t, double v, double Lambda=1,
                        bint isTypeDiff=False):
        cdef incStat incS
        incS = self.register(ID, Lambda, t, isTypeDiff)
        incS.insert(v, t)
        return incS

    # Pulls current stats from the given ID
    cdef list get_1D_Stats(self, str ID, double Lambda=1):  # weight, mean, std
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStat
        cdef incStat incS
        incS = self.HT.get(ID+"_"+str(Lambda))
        if incS is None:  # does not already exist
            # return [np.na]*3
            return []*3
        else:
            return incS.allstats_1D()

    # Pulls current correlational stats from the given IDs
    cdef list get_2D_Stats(self, str ID1, str ID2, double Lambda=1):  # cov, pcc
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStat
        cdef incStat incS
        incS1 = self.HT.get(ID1 + "_" + str(Lambda))
        if incS1 is None:  # does not exist
            # return [np.na]*2
            return []*2

        # find relevant cov entry
        return incS1.cov_pcc(ID2)

    # Pulls all correlational stats registered with the given ID
    # returns tuple [0]: stats-covs&pccs, [2]: IDs
    cdef tuple get_all_2D_Stats(self, str ID, double Lambda=1):  # cov, pcc
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStat
        cdef incStat incS1
        incS1 = self.HT.get(ID + "_" + str(Lambda))
        if incS1 is None:  # does not exist
            return ([], [])

        # find relevant cov entry
        stats = []
        IDs = []
        cdef incStat_cov cov
        for c in incS1.covs:
            cov = c
            stats.append(cov.get_stats1())
            # IDs.append([cov.incStats[0].ID, cov.incStats[1].ID])
            IDs.append([cov.incS1.ID, cov.incS2.ID])
        return stats, IDs

    # Pulls current multidimensional stats from the given IDs
    cdef list get_nD_Stats(self, list IDs, double Lambda=1):  # radius, magnitude (IDs is a list)
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStats
        incStats = []
        for ID in IDs:
            incS = self.HT.get(ID + "_" + str(Lambda))
            if incS is not None:  # exists
                incStats.append(incS)

        # Compute stats
        cdef double rad, mag
        rad = 0  # radius
        mag = 0  # magnitude
        for incS in incStats:
            rad += incS.var()
            mag += incS.mean()**2

        return [sqrt(rad), sqrt(mag)]

    # Updates and then pulls current 1D stats from the given ID.
    # Automatically registers previously unknown stream IDs
    cdef list update_get_1D_Stats(self, str ID, double t, double v,
                                  double Lambda=1, bint isTypeDiff=False):  # weight, mean, std
        cdef incStat incS
        incS = self.update(ID, t, v, Lambda, isTypeDiff)
        return incS.allstats_1D()

    # Updates and then pulls current correlative stats between the given IDs.
    # Automatically registers previously unknown stream IDs, and cov tracking
    # Note: AfterImage does not currently support Diff Type streams for correlational statistics.
    cdef list update_get_2D_Stats(self, str ID1, str ID2, double t1, double v1,
                                  double Lambda=1, int level=1):
        # level= 1:cov,pcc 2:radius,magnitude,cov,pcc
        # retrieve/add cov tracker
        cdef incStat_cov inc_cov
        inc_cov = self.register_cov(ID1, ID2, Lambda,  t1)
        # print(f'inc cov sum res prod before: {inc_cov.CF3}')
        # Update cov tracker
        # inc_cov.update_cov(ID1, v1, t1)
        # print(f'inc cov sum res prod after:  {inc_cov.CF3}')
        # print(ID1)
        # print(ID2)
        # print(f'inc cov S1: {inc_cov.incStats[0].ID}, inc cov S2: {inc_cov.incStats[1].ID}')
        if level == 1:
            return inc_cov.get_stats1()
        else:
            return inc_cov.get_stats2()

    # Updates and then pulls current 1D and 2D stats from the given IDs.
    # Automatically registers previously unknown stream IDs
    cdef list update_get_1D2D_Stats(self, str ID1, str ID2, double t1,
                                    double v1, double Lambda=1):  # weight, mean, std
        return self.update_get_1D_Stats(ID1, t1, v1, Lambda) + self.update_get_2D_Stats(
            ID1, ID2, t1, v1, Lambda, level=2)

    cdef list getHeaders_1D(self, double Lambda=1, str ID=None):
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)
        hdrs = incStat(Lambda, ID).getHeaders_1D(suffix=False)
        return [str(Lambda) + "_" + s for s in hdrs]

    cdef list getHeaders_2D(self, double Lambda=1, list IDs=None, int ver=1):  # IDs is a 2-element list or tuple
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)
        if IDs is None:
            IDs = [0, 1]
        hdrs = incStat_cov(incStat(Lambda, IDs[0]),
                           incStat(Lambda, IDs[1]), Lambda).getHeaders(ver, suffix=False)
        return [str(Lambda) + "_" + s for s in hdrs]

    cdef list getHeaders_1D2D(self, double Lambda=1, list IDs=None, int ver=1):
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)
        if IDs is None:
            IDs = ['0', '1']
        hdrs1D = self.getHeaders_1D(Lambda, IDs[0])
        hdrs2D = self.getHeaders_2D(Lambda, IDs, ver)
        return hdrs1D + hdrs2D

    cdef list getHeaders_nD(self, double Lambda=1, list IDs=[]):  # IDs is a n-element list or tuple
        # Default Lambda?
        ID = ":"
        for s in IDs:
            ID += "_"+s
        Lambda = self.get_lambda(Lambda)
        hdrs = ["radius"+ID, "magnitude"+ID]
        return [str(Lambda)+"_"+s for s in hdrs]

    cdef double clean_out_old_records(self, double cutoffWeight):
        before = len(self.HT)
        self.HT = {k: v for (k, v) in self.HT.items() if v.w > cutoffWeight}
        after = len(self.HT)
        return before - after
