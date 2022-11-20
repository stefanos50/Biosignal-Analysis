import scipy
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import jadeR

# This script demonstrates the use of (two different algorithms of) ICA
# for isolating maternal from foetal cardiac signal
# First the Jade's algorithm is invoked
# and then the FASTICA toolbox is utilized (from command line mode)
# Note: you should add to the path the foetal_ecg_ica_example

def plot(time_axis,signals,title):
    fig = plt.figure()
    total_results = len(signals)
    for i in range(len(signals)):
        plt.subplot(total_results, 1, i+1)
        plt.plot(time_axis,signals[i])
    fig.suptitle(title)
    plt.show()

#read sensors signals
datContent = [i.strip().split() for i in open("G:\\CMR_SPbasics\\foetal_ecg_ica_example\\FOETAL_ECG.dat").readlines()]
df = pd.DataFrame(datContent)
print(df)

time_axis = df.iloc[:, 0]
ecg_sensors = df.iloc[: , 2:5]

#preprocess and plot mixed signals
mixed_signals= ecg_sensors.values.tolist()
mixed_signal_Jade = ecg_sensors.values.tolist()
for i in range(len(mixed_signals)):
    for j in range(len(mixed_signals[i])):
        mixed_signals[i][j] = float(mixed_signals[i][j])
        mixed_signal_Jade[i][j] = float(mixed_signal_Jade[i][j])
mixed_signals = stats.zscore(mixed_signals,ddof=1)
mixed_signals = np.array(mixed_signals)
mixed_signals = mixed_signals.transpose()
plot(time_axis,mixed_signals,"Mixed Signals")

#jade algorithm
B = jadeR.jadeR(np.array(mixed_signal_Jade).transpose()) # deriving the unmixing-matrix
Sources = np.matmul(B,np.array(mixed_signal_Jade).transpose()) #estimating the ICs (source-signal)
Sources = Sources.transpose()
Sources = stats.zscore(Sources.tolist(),ddof=1)
Sources = np.array(Sources)
Sources = Sources.transpose()
plot(time_axis,Sources,"Jade Algorithm")
print(Sources)

#fast ica algorithm
ica = FastICA(2)
S_ = ica.fit_transform(ecg_sensors)  # Get the estimated sources
S_ = stats.zscore(S_,ddof=1)
S_ = np.array(S_)
S_ = S_.transpose()
plot(time_axis,S_,"Fast ICA")
print(S_)