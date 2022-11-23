from scapy.all import rdpcap, IP, IPv6, UDP, TCP, ARP, ICMP
import sys
import os.path
import subprocess
import numpy as np
import pandas as pd
import csv
import netStat as nS

print("Importing Scapy Library")


class FE:
    def __init__(self, file_path, limit=np.inf, fm_grace=100000, ad_grace=900000, train_skip=False):
        self.path = file_path
        self.limit = limit
        self.parse_type = None  # unknown
        self.tsvin = None  # used for parsing TSV file
        self.scapyin = None  # used for parsing pcap with scapy

        if train_skip:
            self.curPacketIndx = fm_grace + ad_grace
        else:
            self.curPacketIndx = 0

        # Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = nS.netStat(np.nan, maxHost, maxSess)

        self.__check_csv__()

    def __check_csv__(self):
        # Check the file type.
        file_path = self.path.split('.')[0]

        if not os.path.isfile(file_path + '.csv'):
            self.parse_pcap()

        self.df_csv = pd.read_csv(file_path + '.csv')
        self.limit = self.df_csv.shape[0]

    def get_next_vector(self, flag):

        if self.curPacketIndx == self.limit:
            return []

        if not flag:
            self.curPacketIndx = self.curPacketIndx + 1
            return [0]

        # Parse next packet ###
        row = self.df_csv.iloc[self.curPacketIndx]
        IPtype = np.nan
        timestamp = row[0]
        framelen = row[1]
        srcIP = ''
        dstIP = ''
        if row[4] != '':  # IPv4
            srcIP = row[4]
            dstIP = row[5]
            IPtype = 0
        elif row[17] != '':  # ipv6
            srcIP = row[17]
            dstIP = row[18]
            IPtype = 1
        # UDP/TCP port: concat of strings will result in an OR "[tcp|udp]"
        srcproto = str(row[6]) + str(row[8])
        dstproto = str(row[7]) + str(row[9])  # UDP or TCP port
        srcMAC = row[2]
        dstMAC = row[3]
        if srcproto == '':  # it's a L2/L1 level protocol
            if row[12] != '':  # is ARP
                srcproto = 'arp'
                dstproto = 'arp'
                srcIP = row[14]  # src IP (ARP)
                dstIP = row[16]  # dst IP (ARP)
                IPtype = 0
            elif row[10] != '':  # is ICMP
                srcproto = 'icmp'
                dstproto = 'icmp'
                IPtype = 0
            # some other protocol
            elif srcIP + srcproto + dstIP + dstproto == '':
                srcIP = row[2]  # src MAC
                dstIP = row[3]  # dst MAC

        self.curPacketIndx = self.curPacketIndx + 1
        srcproto = srcproto.replace('nan', '')[:-2]
        dstproto = dstproto.replace('nan', '')[:-2]

        # Extract Features
        try:
            cur_pkt = [str(srcIP), str(dstIP), str(IPtype), str(srcproto), str(dstproto)]

            cur_pkt_stats = self.nstat.updateGetStats(str(IPtype), str(srcMAC), str(dstMAC), str(srcIP), srcproto, str(dstIP), dstproto, int(framelen), float(timestamp))
            # if self.curPacketIndx == 1000000:
            #     self.nstat.reset_stats()

            return [cur_pkt, cur_pkt_stats]
        except Exception as e:
            print(e)
            return []

    def parse_pcap(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst \
                    -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport \
                    -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code \
                    -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 \
                    -e arp.dst.hw_mac -e arp.dst.proto_ipv4 \
                    -e ipv6.src -e ipv6.dst"
        cmd = 'tshark -r ' + self.path + ' -T fields ' + \
            fields + ' -E separator=\',\' -E header=y -E occurrence=f > ' + self.path.split('.')[0] + ".csv"
        subprocess.call(cmd, shell=True)
        print("tshark parsing complete. File saved as: " + self.path.split('.')[0] + ".csv")

    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())
