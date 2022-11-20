import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from time import time

# A simple script demonstrating the use of notch-filter
#             so as to suppress power-line noise 60 HZ  in ECGtrace
# -it relates to problem 4.19
def time_axis(signal_data,fs):
    time = []
    for i in range(len(signal_data)):
        time.append(i+1)
    return [round((i * (1/fs)),2) for i in time]

def plot_signal(time_axis,data):
    fig, axs = plt.subplots()
    axs.set_title("Signal")
    axs.plot(time_axis, data, color='C0')
    axs.set_xlabel("Time")
    plt.show()


fs = 250.0  # Sample frequency (Hz)
Wn=np.array([58/(fs/2) ,62/(fs/2)]) # Normalized cutoff frequencies
mat = scipy.io.loadmat('G:\\CMR_SPbasics\\ECG_60Hz_data')
signal_data = mat['x'][0]
time_axis = time_axis(signal_data,fs)
plot_signal(time_axis,signal_data)

#design an stop-band FIR 36-order filter for removing powerline nosie @ 60 Hz
n=35
t0 = time()
b = scipy.signal.firwin(n,Wn, window = "hamming", pass_zero = "bandstop")
filtered = signal.filtfilt(b,1, signal_data) #apply the filter in zero-phase (bidirectional) mode
print(f"FIR time: {time() - t0}s")
plot_signal(time_axis,filtered)

#design an stop-band IIR 3-order filter for removing powerline nosie @ 60 Hz
n=3   #Filter_Order
t0 = time()
b, a = signal.butter(n, Wn, btype='stop')
filtered = signal.filtfilt(b, a, signal_data) #apply the filter in zero-phase (bidirectional) mode
print(f"IIR time: {time() - t0}s")
plot_signal(time_axis,filtered)
