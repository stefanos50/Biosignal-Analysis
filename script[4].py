import scipy as scipy
import matplotlib.pyplot as plt
from scipy import signal
from time import time

# script that relates to problem 4.14
#it demonstrates the use of high-pass filtering to suppress noise that
#contaminates ECG-trace and relates to respiration artifact
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
Wn=8/(fs/2) # Normalized cutoff frequencies
mat = scipy.io.loadmat('G:\\CMR_SPbasics\\ECG_9')
signal_data = mat['x'][0]
time_axis = time_axis(signal_data,fs)
plot_signal(time_axis,signal_data)

# design an a high-pass FIR 32-order dilter with cutoff at 8 Hz
n=33
b = scipy.signal.firwin(n,Wn, window = "hamming", pass_zero = "highpass") # apply the filter in causal (forward-direction) mode
t0 = time()
filtered = scipy.signal.lfilter(b,1,signal_data)
print(f"lfilter time: {time() - t0}s")
plot_signal(time_axis,filtered)

t0 = time()
filtered = signal.filtfilt(b,1, signal_data) #apply the filter in zero-phase mode (i.e. bidirectionally)
print(f"filtfilt time: {time() - t0}s")
plot_signal(time_axis,filtered)

