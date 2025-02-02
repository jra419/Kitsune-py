import os
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from Kitsune import Kitsune
from sklearn import metrics

##############################################################################

# Kitsune a lightweight online network intrusion detection system based on
# an ensemble of autoencoders (kitNET).
# For more information and citation, please see our NDSS'18 paper:
# Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates Kitsune's ability to incrementally learn,
# and detect anomalies in recorded a pcap of the Mirai Malware.
# The demo involves an m-by-n dataset with n=115 dimensions (features),
# and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of
# incremental damped statistics (see the NDSS paper for more details)

# The runtimes presented in the paper, are based on the C++ implementation
# (roughly 100x faster than the python implementation)
# ##################  Last Tested with Anaconda 3.6.3   #######################

argparser = argparse.ArgumentParser(description='Kitsune.')
argparser.add_argument('--trace', type=str, help='Trace file path')
argparser.add_argument('--labels', type=str, help='Trace labels file path')
argparser.add_argument('--sampling', type=int, default=1, help='Execution phase sampling rate')
argparser.add_argument('--offset', type=int, default=0, help='Execution phase starting offset')
argparser.add_argument('--fm_grace', type=int, default=100000, help='FM grace period')
argparser.add_argument('--ad_grace', type=int, default=900000, help='AD grace period')
argparser.add_argument('--max_ae', type=int, default=10, help='KitNET: m value')
argparser.add_argument('--train_stats', type=str, default=None, help='Prev. trained stats path')
argparser.add_argument('--fm_model', type=str, help='Prev. trained FM model path')
argparser.add_argument('--el_model', type=str, help='Prev. trained EL path')
argparser.add_argument('--ol_model', type=str, help='Prev. trained OL path')
argparser.add_argument('--attack', type=str, help='Current trace attack name')
args = argparser.parse_args()

labels = pd.read_csv(args.labels, header=None)
ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]
packet_limit = np.Inf

outdir = str(Path(__file__).parents[0]) + '/eval'
if not os.path.exists(str(Path(__file__).parents[0]) + '/eval'):
    os.mkdir(outdir)
outpath = os.path.join(outdir, args.attack + '-' + str(args.sampling) + '-o-' + str(args.offset) + '.csv')

learning_rate = 0.1
hidden_ratio = 0.75

pkt_cnt_global = 0

if args.fm_model is not None and args.el_model is not None and args.ol_model is not None:
    train_skip = True
    trace_row = args.fm_grace + args.ad_grace + args.offset
else:
    train_skip = False
    trace_row = 0

# Build Kitsune
K = Kitsune(args.trace, packet_limit, args.max_ae, args.fm_grace, args.ad_grace, learning_rate,
            hidden_ratio, args.fm_model, args.el_model, args.ol_model, args.train_stats,
            args.attack, train_skip, args.offset)

print("Running Kitsune:")

start = time.time()
RMSEs = []
kitsune_eval = []

# The threshold value is obtained from the highest RMSE score during the training phase.
threshold = 0

old_time = 0
new_time = 0

# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    trace_row += 1
    pkt_cnt_global += 1

    if trace_row % 1000 == 0:
        new_time = time.time()
        print(f'Elapsed time: {new_time - old_time} ({int(1000/(new_time - old_time))} pps)')
        print(trace_row)
        old_time = new_time
    # During the training phase, process all packets.
    # After reaching the execution phase, process w/ sampling.
    if trace_row <= args.fm_grace + args.ad_grace:
        [ pkt, rmse ], framelen, (dt, dt_fe, dt_ad) = K.proc_next_packet(True)
        # [ pkt, rmse ], framelen, (dt, dt_fe, dt_ad), pktstats = K.proc_next_packet(True)
        # if trace_row > 5:
        #     break
    else:
        # At the start of the execution phase, retrieve the highest RMSE score from training.
        if trace_row == args.fm_grace + args.ad_grace + 1 and not train_skip:
            threshold = max(RMSEs, key=float)
            trace_row += args.offset
        if pkt_cnt_global % args.sampling == 0:
            [ pkt, rmse ], framelen, (dt, dt_fe, dt_ad) = K.proc_next_packet(True)
            # [ pkt, rmse ], framelen, (dt, dt_fe, dt_ad), pktstats = K.proc_next_packet(True)
        else:
            [ pkt, rmse ], framelen, (dt, dt_fe, dt_ad) = K.proc_next_packet(False)
            # [ pkt, rmse ], framelen, (dt, dt_fe, dt_ad), pktstats = K.proc_next_packet(False)
    # if trace_row > 1200000:
    #     break
    if rmse == -1:
        break
    # if rmse == 0:
    #     continue
    RMSEs.append(rmse)
    try:
        kitsune_eval.append([pkt[0], pkt[1], pkt[2], pkt[3], pkt[4],
                            rmse, labels.iloc[trace_row - 1][0]])
        # kitsune_eval.append([rmse, labels.iloc[trace_row - 1][0]])
        # with open(outpath, "a") as f:
        #     f.write(f'{pkt[0]},{pkt[1]},{pkt[2]},{pkt[3]},{pkt[4]},{rmse},{labels.iloc[trace_row -1][0]},{pktstats}')
    except IndexError:
        print('pkt_cnt_global: ' + str(pkt_cnt_global))

stop = time.time()
print('Complete. Time elapsed: ' + str(stop - start))
print('Threshold: ' + str(threshold))

# Collect the processed packets' RMSE, label, and save to a csv.
df_kitsune = pd.DataFrame(kitsune_eval,
                          columns=['ip_src', 'ip_dst', 'ip_type', 'src_proto',
                                   'dst_proto', 'rmse', 'label'])
                          # columns=['rmse', 'label'])
df_kitsune.to_csv(outpath, chunksize=10000, index=None)

# Cut all training rows.
# if train_skip is False:
#     df_kitsune_cut = df_kitsune.drop(df_kitsune.index[range(args.fm_grace + args.ad_grace)])
# else:
#     df_kitsune_cut = df_kitsune
df_kitsune_cut = df_kitsune

# Sort by RMSE.
df_kitsune_cut.sort_values(by='rmse', ascending=False, inplace=True)

# Split by threshold.
kitsune_benign = df_kitsune_cut[df_kitsune_cut.rmse < threshold]
print(kitsune_benign.shape[0])
kitsune_alert = df_kitsune_cut[df_kitsune_cut.rmse >= threshold]
print(kitsune_alert.shape[0])

# Calculate statistics.
TP = kitsune_alert[kitsune_alert.label == 1].shape[0]
FP = kitsune_alert[kitsune_alert.label == 0].shape[0]
TN = kitsune_benign[kitsune_benign.label == 0].shape[0]
FN = kitsune_benign[kitsune_benign.label == 1].shape[0]

try:
    TPR = TP / (TP + FN)
except ZeroDivisionError:
    TPR = 0

try:
    TNR = TN / (TN + FP)
except ZeroDivisionError:
    TNR = 0

try:
    FPR = FP / (FP + TN)
except ZeroDivisionError:
    FPR = 0

try:
    FNR = FN / (FN + TP)
except ZeroDivisionError:
    FNR = 0

try:
    accuracy = (TP + TN) / (TP + FP + FN + TN)
except ZeroDivisionError:
    accuracy = 0

try:
    precision = TP / (TP + FP)
except ZeroDivisionError:
    precision = 0

try:
    recall = TP / (TP + FN)
except ZeroDivisionError:
    recall = 0

try:
    f1_score = 2 * (recall * precision) / (recall + precision)
except ZeroDivisionError:
    f1_score = 0

roc_curve_fpr, roc_curve_tpr, roc_curve_thres = metrics.roc_curve(df_kitsune_cut.label,
                                                                  df_kitsune_cut.rmse)
roc_curve_fnr = 1 - roc_curve_tpr

auc = metrics.roc_auc_score(df_kitsune_cut.label, df_kitsune_cut.rmse)
eer = roc_curve_fpr[np.nanargmin(np.absolute((roc_curve_fnr - roc_curve_fpr)))]
eer_sanity = roc_curve_fnr[np.nanargmin(np.absolute((roc_curve_fnr - roc_curve_fpr)))]

print('TP: ' + str(TP))
print('TN: ' + str(TN))
print('FP: ' + str(FP))
print('FN: ' + str(FN))
print('TPR: ' + str(TPR))
print('TNR: ' + str(TNR))
print('FPR: ' + str(FPR))
print('FNR: ' + str(FNR))
print('Accuracy: ' + str(accuracy))
print('precision: ' + str(precision))
print('Recall: ' + str(recall))
print('F1 Score: ' + str(f1_score))
print('AuC: ' + str(auc))
print('EER: ' + str(eer))
print('EER sanity: ' + str(eer_sanity))

# Write the eval to a txt.
f = open('eval/' + args.attack + '-' + str(args.sampling) + '-o-' + str(args.offset) + '.txt', 'a+')
f.write('Time elapsed: ' + str(stop - start) + '\n')
f.write('Threshold: ' + str(threshold) + '\n')
f.write('TP: ' + str(TP) + '\n')
f.write('TN: ' + str(TN) + '\n')
f.write('FP: ' + str(FP) + '\n')
f.write('FN: ' + str(FN) + '\n')
f.write('TPR: ' + str(TPR) + '\n')
f.write('TNR: ' + str(TNR) + '\n')
f.write('FPR: ' + str(FPR) + '\n')
f.write('FNR: ' + str(FNR) + '\n')
f.write('Accuracy: ' + str(accuracy) + '\n')
f.write('Precision: ' + str(precision) + '\n')
f.write('Recall: ' + str(recall) + '\n')
f.write('F1 Score: ' + str(f1_score) + '\n')
f.write('AuC: ' + str(auc) + '\n')
f.write('EER: ' + str(eer) + '\n')
f.write('EER sanity: ' + str(eer_sanity) + '\n')
