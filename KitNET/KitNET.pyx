import os
import numpy as np
cimport numpy as cnp
import pickle
from pathlib import Path
cimport KitNET.dA as AE
cimport KitNET.corClust as CC

cnp.import_array()

# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper:
# Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document

DTYPE = np.double

cdef class KitNET:
    # n: the number of features in your input dataset (i.e., x \in R^n)
    # m: the maximum size of any autoencoder in the ensemble layer
    # AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    # FM_grace_period: the number of instances which will be taken to learn the feature mapping.
    # If 'None', then FM_grace_period=AM_grace_period
    # learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    # hidden_ratio: the default ratio of hidden to visible neurons.
    # E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    # feature_map: One may optionally provide a feature map instead of learning one.
    # The map must be a list, where the i-th entry contains a list of the feature indices
    # to be assingned to the i-th autoencoder in the ensemble.
    # For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(self, int n, int max_autoencoder_size=10, int FM_grace_period=10000,
                 int AD_grace_period=90000, double learning_rate=0.1, double hidden_ratio=0.75,
                 str feature_map=None, str ensemble_layer=None, str output_layer=None,
                 str attack=''):
        # Parameters:
        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n

        # Variables
        self.n_trained = 0  # the number of training instances so far
        self.n_executed = 0  # the number of executed instances so far
        self.ensemble_layer = []
        self.output_layer = None
        self.attack = attack

        # Check if the feature map, ensemble layer and output layer are provided as input.
        # If so, skip the training phase.

        if feature_map is not None:
            with open(feature_map, 'rb') as f_fm:
                self.v = pickle.load(f_fm)
            self.createAD()
            print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode", flush=True)
        else:
            self.v = None
            print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode", flush=True)
        self.FM = CC.corClust(self.n)  # incremental feature clustering for the feature mapping process

        if ensemble_layer is not None and output_layer is not None:
            with open(ensemble_layer, 'rb') as f_el:
                self.ensemble_layer = pickle.load(f_el)
            with open(output_layer, 'rb') as f_ol:
                self.output_layer = pickle.load(f_ol)
            self.n_trained = self.FM_grace_period + self.AD_grace_period + 1
            print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode", flush=True)

    # If FM_grace_period+AM_grace_period has passed, then this function executes KitNET on x.
    # Otherwise, this function learns from x. x: a numpy array of length n
    # Note: KitNET automatically performs 0-1 normalization on all attributes.
    cdef double process(self, cnp.ndarray[DTYPE_t, ndim=1] x):
        # If both the FM and AD are in execute-mode
        if self.n_trained >= self.FM_grace_period + self.AD_grace_period:
            return self.execute(x)
        else:
            return self.train(x)

    # force train KitNET on x
    # returns the anomaly score of x during training (do not use for alerting)
    cdef double train(self, cnp.ndarray[DTYPE_t, ndim=1] x):
        cdef cnp.ndarray[DTYPE_t, ndim=1] S_l1
        cdef cnp.ndarray[DTYPE_t, ndim=1] xi
        cdef double output
        cdef str outdir
        if self.n_trained < self.FM_grace_period and self.v is None:
            # If the FM is in train-mode, and the user has not supplied a feature mapping
            # update the incremetnal correlation matrix
            self.FM.update(x)
            # If the feature mapping should be instantiated
            if self.n_trained == self.FM_grace_period - 1:
                self.v = self.FM.cluster(self.m)
                self.createAD()
                print("The Feature-Mapper found a mapping: " + str(self.n) + " features to " + str(
                    len(self.v))+" autoencoders.")
                print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
            self.n_trained += 1
            return 0.0
        else:  # train
            # Ensemble Layer
            S_l1 = np.zeros(len(self.ensemble_layer), dtype=DTYPE)
            for a in range(len(self.ensemble_layer)):
                # make sub instance for autoencoder 'a'
                xi = x[self.v[a]]
                S_l1[a] = self.ensemble_layer[a].train(xi)
            # OutputLayer
            output = self.output_layer.train(S_l1)
            if self.n_trained == self.AD_grace_period+self.FM_grace_period - 1:
                print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
                outdir = str(Path(__file__).parents[0]) + '/models'
                if not os.path.exists(str(Path(__file__).parents[0]) + '/models'):
                    os.mkdir(outdir)

                with open(outdir + '/' + self.attack + '-m-' + str(self.m) + '-fm' + '.txt', 'wb') as f_fm:
                    pickle.dump(self.v, f_fm)
                with open(outdir + '/' + self.attack + '-m-' + str(self.m) + '-el' + '.txt', 'wb') as f_el:
                    pickle.dump(self.ensemble_layer, f_el)
                with open(outdir + '/' + self.attack + '-m-' + str(self.m) + '-ol' + '.txt', 'wb') as f_ol:
                    pickle.dump(self.output_layer, f_ol)
            self.n_trained += 1
            return output

    # force execute KitNET on x
    cdef double execute(self, cnp.ndarray[DTYPE_t, ndim=1] x):
        cdef cnp.ndarray[DTYPE_t, ndim=1] S_l1
        cdef cnp.ndarray[DTYPE_t, ndim=1] xi
        if self.v is None:
            raise RuntimeError(
                'KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running '
                'process(x) instead.')
        else:
            self.n_executed += 1
            # Ensemble Layer
            S_l1 = np.zeros(len(self.ensemble_layer), dtype=DTYPE)
            for a in range(len(self.ensemble_layer)):
                # make sub inst
                xi = x[self.v[a]]
                S_l1[a] = self.ensemble_layer[a].execute(xi)
            # OutputLayer
            return self.output_layer.execute(S_l1)

    cdef void createAD(self):
        # construct ensemble layer
        cdef AE.dA_params params
        cdef AE.dA da
        for ad_map in self.v:
            params = AE.dA_params(n_visible=len(ad_map), n_hidden=0, lr=self.lr,
                                  corruption_level=0, grace_period=0, hidden_ratio=self.hr)
            da = AE.dA(params)
            self.ensemble_layer.append(da)
        # construct output layer
        params = AE.dA_params(len(self.v), n_hidden=0, lr=self.lr, corruption_level=0,
                              grace_period=0, hidden_ratio=self.hr)
        self.output_layer = AE.dA(params)
