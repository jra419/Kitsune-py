from FeatureExtractor import FE
from KitNET.KitNET import KitNET

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


class Kitsune:
    def __init__(self, file_path, limit, max_autoencoder_size=10, fm_grace=None,
                 ad_grace=10000, learning_rate=0.1, hidden_ratio=0.75, feature_map=None,
                 ensemble_layer=None, output_layer=None, train_stats=None, attack='',
                 train_skip=False):

        # init packet feature extractor (AfterImage)
        self.FE = FE(file_path, limit, max_autoencoder_size, fm_grace, ad_grace,
                     train_stats, train_skip, attack)

        # init Kitnet
        self.AnomDetector = KitNET(self.FE.get_num_features(), max_autoencoder_size, fm_grace,
                                   ad_grace, learning_rate, hidden_ratio, feature_map,
                                   ensemble_layer, output_layer, attack)

    def proc_next_packet(self, flag):
        x = self.FE.get_next_vector(flag)

        # If the trace has ended, return -1.
        # Else if this cur packet is to be processed due to sampling, proceed to the classifier.
        # Else, skip the packet classification and return 0.
        if len(x) == 0:
            return -1,-1
        elif flag:
            return [x[0], self.AnomDetector.process(x[1])],x[2]
        else:
            return 0,x[2]
