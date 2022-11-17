import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
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

# File location
path = str(sys.argv[1])  # the pcap, pcapng, or tsv file to process.
path_labels = str(sys.argv[2])
packet_div = int(sys.argv[3])
attack_name = str(sys.argv[4])
packet_limit = np.Inf  # the number of packets to process

labels = pd.read_csv(path_labels, header=None)
ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]

# KitNET params:
maxAE = 10  # maximum size for any autoencoder in the ensemble layer
FMgrace = 100000  # the number of instances taken to learn the feature mapping
ADgrace = 900000  # the number of instances used to train the anomaly detector

outdir = str(Path(__file__).parents[0]) + '/eval'
if not os.path.exists(str(Path(__file__).parents[0]) + '/eval'):
    os.mkdir(outdir)
outpath = os.path.join(outdir, attack_name + '.csv')

# Build Kitsune
K = Kitsune(path, packet_limit, maxAE, FMgrace, ADgrace)

print("Running Kitsune:")

start = time.time()
RMSEs = []
kitsune_eval = []
trace_row = 0
label_row = 0

# The threshold value is obtained from the highest RMSE score during the training phase.
threshold = 0

# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    trace_row += 1
    if trace_row % 1000 == 0:
        print(trace_row)
    # During the training phase, process all packets.
    # After reaching the execution phase, process w/ sampling.
    if trace_row <= FMgrace+ADgrace:
        rmse = K.proc_next_packet(True)
    else:
        # At the start of the execution phase, retrieve the highest RMSE score from training.
        if trace_row == FMgrace+ADgrace+1:
            threshold = max(RMSEs, key=float)
        if trace_row % packet_div == 0:
            rmse = K.proc_next_packet(True)
        else:
            rmse = K.proc_next_packet(False)
    if rmse == -1:
        break
    if rmse == 0:
        continue
    RMSEs.append(rmse[1])
    try:
        kitsune_eval.append([rmse[0][0], rmse[0][1], rmse[0][2], rmse[0][3], rmse[0][4],
                            rmse[1], labels.iloc[trace_row - 1][0]])
    except IndexError:
        print('trace_row: ' + str(trace_row))
        print('label_row: ' + str(label_row))
    label_row += 1

stop = time.time()
print('Complete. Time elapsed: ' + str(stop - start))
print('Threshold: ' + str(threshold))

# Collect the processed packets' RMSE, label, and save to a csv.
df_kitsune = pd.DataFrame(kitsune_eval,
                          columns=['ip_src', 'ip_dst', 'ip_type', 'src_proto', 'dst_proto', 'rmse', 'label'])
df_kitsune.to_csv(outpath, index=None)

# Cut all training rows.
df_kitsune_cut = df_kitsune.drop(df_kitsune.index[range(FMgrace+ADgrace)])

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

roc_curve_fpr, roc_curve_tpr, roc_curve_thres = metrics.roc_curve(df_kitsune.label, df_kitsune.rmse)
roc_curve_fnr = 1 - roc_curve_tpr

auc = metrics.roc_auc_score(df_kitsune.label, df_kitsune.rmse)
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
f = open('eval/' + attack_name + '.txt', 'a+')
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

# Plot the RMSE anomaly scores.
# print("Plotting results")
# plt.figure(figsize=(10, 5))
# cmap = colors.ListedColormap(['green', 'red'])
# bounds = [0, threshold, max(RMSEs[FMgrace+ADgrace+1:])]
# norm = colors.BoundaryNorm(bounds, cmap.N)
# cmap.set_over('red')
# cmap.set_under('green')
# fig = plt.scatter(range(FMgrace+ADgrace+1, len(RMSEs)),
#                   RMSEs[FMgrace+ADgrace+1:],
#                   s=0.1,
#                   c=RMSEs[FMgrace+ADgrace+1:],
#                   cmap=cmap,
#                   norm=norm,
#                   vmin=threshold,
#                   vmax=threshold)
# plt.yscale("log")
# plt.title("Anomaly Scores from Kitsune's Execution Phase")
# plt.ylabel("RMSE (log scaled)")
# plt.xlabel("Packets")
# plt.hlines(xmin=0, xmax=len(RMSEs), y=threshold, colors='blue', label=threshold)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3))
# plt.subplots_adjust(bottom=0.25)
# figbar = plt.colorbar()
# plt.savefig('eval/' + attack_name + '.png')
# plt.show()
