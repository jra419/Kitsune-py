import os.path
import subprocess
import numpy as np
import pandas as pd
import netStat as nS
from math import isnan

class FE:
    def __init__(self, file_path, limit=np.inf, max_autoencoder_size=10,
                 fm_grace=100000, ad_grace=900000, train_stats=None,
                 train_skip=False, attack='', offset=0):

        self.path = file_path
        self.limit = limit
        self.parse_type = None  # unknown
        self.tsvin = None  # used for parsing TSV file
        self.scapyin = None  # used for parsing pcap with scapy
        self.fm_grace = fm_grace
        self.ad_grace = ad_grace
        self.train_skip = train_skip
        self.offset = offset

        if train_skip:
            self.curPacketIndx = self.fm_grace + self.ad_grace + self.offset
        else:
            self.curPacketIndx = 0

        # Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = nS.netStat(np.nan, maxHost, maxSess, max_autoencoder_size,
                                train_stats, train_skip, attack)

        self.__check_csv__()

    def __check_csv__(self):
        # Check the file type.
        file_path = self.path.split('.')[0]

        if not os.path.isfile(file_path + '.csv'):
            self.parse_pcap()

        self.df_csv = pd.read_csv(file_path + '.csv')
        self.limit = self.df_csv.shape[0]

    def get_next_vector(self, flag):

        if not self.train_skip and self.curPacketIndx == self.fm_grace + self.ad_grace:
            self.curPacketIndx += self.offset

        if self.curPacketIndx == self.limit:
            return [ -1, -1, -1 ]


        if not flag:
            self.curPacketIndx = self.curPacketIndx + 1
            return [ 0, 0, 0 ]


        # Parse next packet ###
        IPtype = np.nan
        timestamp = self.df_csv.iat[self.curPacketIndx, 0]
        framelen = self.df_csv.iat[self.curPacketIndx, 6]
        if isnan(framelen):
            framelen = 0
        srcIP = ''
        dstIP = ''
        srcIP = self.df_csv.iat[self.curPacketIndx, 4]
        srcIPv6 = self.df_csv.iat[self.curPacketIndx, 19]
        if srcIP != '':  # IPv4
            dstIP = self.df_csv.iat[self.curPacketIndx, 5]
            # IPtype = 0
        elif srcIPv6 != '':  # ipv6
            srcIP = srcIPv6
            dstIP = self.df_csv.iat[self.curPacketIndx, 20]
            # IPtype = 1
        IPtype = self.df_csv.iat[self.curPacketIndx, 7]
        if isnan(IPtype):
            IPtype = 0
        # UDP/TCP port: concat of strings will result in an OR "[tcp|udp]"
        srcproto = str(self.df_csv.iat[self.curPacketIndx, 8]) + \
            str(self.df_csv.iat[self.curPacketIndx, 10])
        dstproto = str(self.df_csv.iat[self.curPacketIndx, 9]) + \
            str(self.df_csv.iat[self.curPacketIndx, 11])  # UDP or TCP port
        srcMAC = self.df_csv.iat[self.curPacketIndx, 2]
        dstMAC = self.df_csv.iat[self.curPacketIndx, 3]
        if srcproto == '':  # it's a L2/L1 level protocol
            arp = self.df_csv.iat[self.curPacketIndx, 14]
            icmp = self.df_csv.iat[self.curPacketIndx, 12]
            if arp != '':  # is ARP
                srcproto = 'arp'
                dstproto = 'arp'
                srcIP = self.df_csv.iat[self.curPacketIndx, 16]  # src IP (ARP)
                dstIP = self.df_csv.iat[self.curPacketIndx, 18]  # dst IP (ARP)
                IPtype = 0
            elif icmp != '':  # is ICMP
                srcproto = 'icmp'
                dstproto = 'icmp'
                IPtype = 0
            # some other protocol
            elif srcIP + srcproto + dstIP + dstproto == '':
                srcIP = srcMAC  # src MAC
                dstIP = dstMAC  # dst MAC

        self.curPacketIndx = self.curPacketIndx + 1
        srcproto = srcproto.replace('nan', '')[:-2]
        dstproto = dstproto.replace('nan', '')[:-2]


        # Extract Features
        try:
            cur_pkt = [str(srcIP), str(dstIP), str(IPtype),
                       str(srcproto), str(dstproto)]
            cur_pkt_stats = self.nstat.updateGetStats(str(IPtype), str(srcMAC),
                                                      str(dstMAC), str(srcIP),
                                                      srcproto, str(dstIP),
                                                      dstproto, int(framelen),
                                                      float(timestamp))
            # if not self.train_skip and self.curPacketIndx == \
            #         self.fm_grace + self.ad_grace:
            #     self.nstat.save_stats()

            return [cur_pkt, cur_pkt_stats, framelen]
        except Exception as e:
            print(e)
            print(self.curPacketIndx)
            print(cur_pkt)
            print(str(IPtype), str(srcMAC), str(dstMAC), str(srcIP), srcproto,
                  str(dstIP), dstproto, int(framelen), float(timestamp))
            return [ -1, -1, -1 ]

    def parse_pcap(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst \
                    -e ip.src -e ip.dst -e ip.len -e ip.proto -e tcp.srcport \
                    -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type \
                    -e icmp.code -e arp.opcode -e arp.src.hw_mac \
                    -e arp.src.proto_ipv4 -e arp.dst.hw_mac \
                    -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd = 'tshark -r ' + self.path + ' -T fields ' + \
            fields + ' -E separator=\',\' -E header=y -E occurrence=f > ' + \
            self.path.split('.')[0] + ".csv"
        subprocess.call(cmd, shell=True)
        print("tshark parsing complete. File saved as: " +
              self.path.split('.')[0] + ".csv")

    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())
