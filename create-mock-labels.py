#!/usr/bin/env python3

import os
import argparse
import subprocess

SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))
LABELS_FILE = f'{SCRIPT_DIR}/labels-mock.txt'

def get_packets_in_pcap(pcap):
    cmd = [ 'capinfos', '-c', '-M', pcap ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    out = proc.stdout.decode('utf-8')
    err = proc.stderr.decode('utf-8')

    if proc.returncode != 0:
        print(err)
        exit(1)
    
    lines = out.split('\n')
    assert len(lines) >= 2

    for line in lines:
        if 'Number of packets' not in line:
            continue
        
        nr_pkts = line.split(' ')[-1]
        nr_pkts = int(nr_pkts)
        
        return nr_pkts
    
    assert False and "Line with number of packets not found."

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Create mock labels file for pcap.')
    argparser.add_argument('pcap', type=str, help='Target pcap')
    args = argparser.parse_args()

    n_pkts = get_packets_in_pcap(args.pcap)

    with open(LABELS_FILE, 'w') as f:
        for i in range(n_pkts):
            f.write('0\n')