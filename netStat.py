import sys
import os
import numpy as np
import AfterImage as af
import pickle
import jsonpickle
from pathlib import Path

#
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    # Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas=np.nan, HostLimit=255, HostSimplexLimit=1000,
                 m=10, train_stats=None, train_skip=False, attack=''):

        sys.setrecursionlimit(100000)

        self.attack = attack
        self.m = m

        # Lambdas
        if np.isnan(Lambdas):
            self.Lambdas = [5, 3, 1, .1, .01]
        else:
            self.Lambdas = Lambdas

        # HT Limits

        self.HostLimit = HostLimit
        # *2 since each dual creates 2 entries in memory
        self.SessionLimit = HostSimplexLimit*self.HostLimit*self.HostLimit
        self.MAC_HostLimit = self.HostLimit*10

        self.HT_jit = af.incStatDB(limit=self.HostLimit*self.HostLimit)  # H-H Jitter Stats
        self.HT_MI = af.incStatDB(limit=self.MAC_HostLimit)  # MAC-IP relationships
        self.HT_H = af.incStatDB(limit=self.HostLimit)  # Source Host BW Stats
        self.HT_Hp = af.incStatDB(limit=self.SessionLimit)  # Source Host BW Stats

        self.HT_jit = af.incStatDB(limit=self.HostLimit*self.HostLimit)  # H-H Jitter Stats
        self.HT_MI = af.incStatDB(limit=self.MAC_HostLimit)  # MAC-IP relationships
        self.HT_H = af.incStatDB(limit=self.HostLimit)  # Source Host BW Stats
        self.HT_Hp = af.incStatDB(limit=self.SessionLimit)  # Source Host BW Stats

        # HTs
        if train_skip:
            with open(train_stats + '-jit.txt', 'rb') as f_stats:
                self.HT_jit.HT = jsonpickle.decode(pickle.load(f_stats))
            with open(train_stats + '-mi.txt', 'rb') as f_stats:
                self.HT_MI.HT = jsonpickle.decode(pickle.load(f_stats))
            with open(train_stats + '-h.txt', 'rb') as f_stats:
                self.HT_H.HT = jsonpickle.decode(pickle.load(f_stats))
            with open(train_stats + '-hp.txt', 'rb') as f_stats:
                self.HT_Hp.HT = jsonpickle.decode(pickle.load(f_stats))

    # cpp: this is all given to you in the direction string of the instance
    # (NO NEED FOR THIS FUNCTION)
    def findDirection(self, IPtype, srcIP, dstIP, eth_src, eth_dst):
        if IPtype == 0:  # is IPv4
            lstP = srcIP.rfind('.')
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind('.')
            dst_subnet = dstIP[0:lstP:]
        elif IPtype == 1:  # is IPv6
            src_subnet = srcIP[0:round(len(srcIP)/2):]
            dst_subnet = dstIP[0:round(len(dstIP)/2):]
        else:  # no Network layer, use MACs
            src_subnet = eth_src
            dst_subnet = eth_dst

        return src_subnet, dst_subnet

    def updateGetStats(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol,
                       dstIP, dstProtocol, datagramSize, timestamp):
        # Host BW: Stats on the srcIP's general Sender Statistics
        # Hstat = np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])

        # MAC.IP: Stats on src MAC-IP relationships
        MIstat = np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            MIstat[(i*3):((i+1)*3)] = self.HT_MI.update_get_1D_Stats(srcMAC+srcIP, timestamp,
                                                                     datagramSize, self.Lambdas[i])
        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HHstat = np.zeros((7*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat[(i*7):((i+1)*7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP, timestamp,
                                                                      datagramSize, self.Lambdas[i])
        # Host-Host Jitter:
        HHstat_jit = np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat_jit[(i*3):((i+1)*3)] = self.HT_jit.update_get_1D_Stats(srcIP+dstIP, timestamp, 0,
                                                                          self.Lambdas[i],
                                                                          isTypeDiff=True)
        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HpHpstat = np.zeros((7*len(self.Lambdas,)))
        if srcProtocol == 'arp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC,
                                                                             timestamp,
                                                                             datagramSize,
                                                                             self.Lambdas[i])
        else:  # some other protocol (e.g. TCP/UDP)
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol,
                                                                             dstIP + dstProtocol,
                                                                             timestamp,
                                                                             datagramSize,
                                                                             self.Lambdas[i])
        # concatenation of stats into one stat vector
        return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat))

    def getNetStatHeaders(self):
        MIstat_headers = []
        Hstat_headers = []
        HHstat_headers = []
        HHjitstat_headers = []
        HpHpstat_headers = []

        for i in range(len(self.Lambdas)):
            MIstat_headers += ["MI_dir_" + h for h in self.HT_MI.getHeaders_1D(
                Lambda=self.Lambdas[i], ID=None)]
            HHstat_headers += ["HH_" + h for h in self.HT_H.getHeaders_1D2D(
                Lambda=self.Lambdas[i], IDs=None, ver=2)]
            HHjitstat_headers += ["HH_jit_" + h for h in self.HT_jit.getHeaders_1D(
                Lambda=self.Lambdas[i], ID=None)]
            HpHpstat_headers += ["HpHp_" + h for h in self.HT_Hp.getHeaders_1D2D(
                Lambda=self.Lambdas[i], IDs=None, ver=2)]
        return MIstat_headers + Hstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers

    def save_stats(self):
        outdir = str(Path(__file__).parents[0]) + '/KitNET/models'

        if not os.path.exists(str(Path(__file__).parents[0]) + '/KitNET/models'):
            os.mkdir(outdir)

        with open(outdir + '/' + self.attack + '-m-' + str(self.m) + '-train-stats-jit' + '.txt', 'wb') as f_stats:
            pickle.dump(jsonpickle.encode(self.HT_jit.HT), f_stats)
        with open(outdir + '/' + self.attack + '-m-' + str(self.m) + '-train-stats-mi' + '.txt', 'wb') as f_stats:
            pickle.dump(jsonpickle.encode(self.HT_MI.HT), f_stats)
        with open(outdir + '/' + self.attack + '-m-' + str(self.m) + '-train-stats-h' + '.txt', 'wb') as f_stats:
            pickle.dump(jsonpickle.encode(self.HT_H.HT), f_stats)
        with open(outdir + '/' + self.attack + '-m-' + str(self.m) + '-train-stats-hp' + '.txt', 'wb') as f_stats:
            pickle.dump(jsonpickle.encode(self.HT_Hp.HT), f_stats)

    def reset_stats(self):
        print('Reset stats')

        self.HT_jit = af.incStatDB(limit=self.HostLimit*self.HostLimit)
        self.HT_MI = af.incStatDB(limit=self.MAC_HostLimit)
        self.HT_H = af.incStatDB(limit=self.HostLimit)
        self.HT_Hp = af.incStatDB(limit=self.SessionLimit)
