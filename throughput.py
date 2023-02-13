import os
import time
import argparse
from pathlib import Path
import numpy as np
from Kitsune import Kitsune

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Kitsune.')
    argparser.add_argument('--trace', type=str, help='Trace file path')
    argparser.add_argument('--sampling', type=int, default=1, help='Execution phase sampling rate')
    argparser.add_argument('--num_pkts', type=int, default=0, help='Maximum number of trace packets to process in execution')
    argparser.add_argument('--fm_grace', type=int, default=100000, help='FM grace period')
    argparser.add_argument('--ad_grace', type=int, default=900000, help='AD grace period')
    argparser.add_argument('--max_ae', type=int, default=10, help='KitNET: m value')
    argparser.add_argument('--train_stats', type=str, default=None, help='Prev. trained stats path')
    argparser.add_argument('--fm_model', type=str, help='Prev. trained FM model path')
    argparser.add_argument('--el_model', type=str, help='Prev. trained EL path')
    argparser.add_argument('--ol_model', type=str, help='Prev. trained OL path')
    argparser.add_argument('--attack', type=str, help='Current trace attack name')
    args = argparser.parse_args()

    packet_limit = np.Inf

    learning_rate = 0.1
    hidden_ratio = 0.75

    if args.fm_model is not None and args.el_model is not None and args.ol_model is not None:
        train_skip = True
        trace_row = args.fm_grace + args.ad_grace
    else:
        train_skip = False
        trace_row = 0

    # Build Kitsune
    K = Kitsune(args.trace, packet_limit, args.max_ae, args.fm_grace, args.ad_grace, learning_rate,
                hidden_ratio, args.fm_model, args.el_model, args.ol_model, args.train_stats,
                args.attack, train_skip)

    print("Running Kitsune", flush=True)

    old_time = 0
    new_time = 0

    pkt_cnt_global       = 0
    pkt_cnt_execution    = 0
    training_start_time  = time.time()
    execution_start_time = -1

    # Here we process (train/execute) each individual packet.
    # In this way, each observation is discarded after performing process() method.
    while True:
        trace_row += 1
        pkt_cnt_global += 1

        if trace_row % 1000 == 0:
            new_time = time.time()
            # print(f'Elapsed time: {new_time - old_time} ({int(1000/(new_time - old_time))} pps)')
            # print(trace_row)
            old_time = new_time

        # During the training phase, process all packets.
        # After reaching the execution phase, process w/ sampling.
        if trace_row <= args.fm_grace + args.ad_grace:
            rmse = K.proc_next_packet(True)
        else:
            # At the start of the execution phase, retrieve the highest RMSE score from training.
            if execution_start_time == -1:
                execution_start_time = time.time()
            
            if args.num_pkts > 0 and pkt_cnt_execution > args.num_pkts:
                break
                
            if pkt_cnt_global % args.sampling == 0:
                rmse = K.proc_next_packet(True)
            else:
                rmse = K.proc_next_packet(False)
            
            pkt_cnt_execution += 1

        if rmse == -1:
            break

        if rmse == 0:
            continue

    stop = time.time()

    training_time   = stop - training_start_time
    execution_time  = stop - execution_start_time
    processing_rate = int(pkt_cnt_execution / execution_time)

    print(f'Training time:   {training_time} s')
    print(f'Execution time:  {execution_time} s')
    print(f'Processing rate: {processing_rate} pps')

    OUT_FILE = f'{args.attack}.csv'
    with open(OUT_FILE, 'w') as f:
        f.write(f'{training_time},{execution_time},{processing_rate}\n')
