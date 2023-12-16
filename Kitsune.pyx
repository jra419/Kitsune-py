cimport FeatureExtractor as FE
cimport KitNET.KitNET as KitNET
from KitNET.KitNET import KitNET

import time

# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


cdef class Kitsune:
    cdef FE.FE FE
    cdef KitNET.KitNET AnomDetector
    def __init__(self, str file_path, double limit, int max_autoencoder_size=10, int fm_grace=10000,
                 int ad_grace=90000, double learning_rate=0.1, double hidden_ratio=0.75,
                 str feature_map=None, str ensemble_layer=None, str output_layer=None,
                 str train_stats=None, str attack='', bint train_skip=False, int offset=0):
        # init packet feature extractor (AfterImage)
        self.FE = FE.FE(file_path, limit, max_autoencoder_size, fm_grace, ad_grace,
                     train_stats, train_skip, attack, offset)

        # init Kitnet
        self.AnomDetector = KitNET(self.FE.get_num_features(), max_autoencoder_size, fm_grace,
                                   ad_grace, learning_rate, hidden_ratio, feature_map,
                                   ensemble_layer, output_layer, attack)

    def proc_next_packet(self, flag):
        start = time.time()

        pkt, pktstats, pktlen = self.FE.get_next_vector(flag)

        fe_end = time.time()

        # If the trace has ended, return -1.
        # Else if this cur packet is to be processed due to sampling, proceed to the classifier.
        # Else, skip the packet classification and return 0.
        if pkt == -1:
            rmse = -1
        elif flag:
            rmse = self.AnomDetector.process(pktstats)
        else:
            rmse = 0

        ad_end = time.time()

        dt    = ad_end - start
        dt_fe = fe_end - start
        dt_ad = ad_end - fe_end

        return [ pkt, rmse ], pktlen, (dt, dt_fe, dt_ad)
        # return [ pkt, rmse ], pktlen, (dt, dt_fe, dt_ad), pktstats
