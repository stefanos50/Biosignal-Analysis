import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import numpy as np
from scipy import signal

mat = scipy.io.loadmat('G:\\CMR_SPbasics\\eeg_data.mat')
eeg = mat['eeg'][0]
fs = 50; #50 hz

def time_axis(signal_data,fs):
    time = []
    for i in range(len(signal_data)):
        time.append(i+1)
    return [round((i * (1/fs)),2) for i in time]

def show_plot(x_axis_data,y_axis_data,x_axis_label,y_axis_label,plot_title,label,bounds):
    plt.plot(x_axis_data, y_axis_data,label=label);
    plt.xlabel(x_axis_label);
    plt.ylabel(y_axis_label);
    plt.title(plot_title);
    if label:
        plt.legend(loc="upper right")
    if bounds:
        plt.xlim(bounds);
    plt.show()

def round_list(ls,decimal):
    for i in range(len(ls)):
        ls[i] = round(ls[i],decimal)
    return ls

def mirror_signal_remove(data):
    tmp = []
    for i in range(round(len(data)/2+1)):
        tmp.append(data[i])
    return tmp

def frequency_domain_f(Frequency,L):
    f = []
    for i in range(0,round(L/2)+2):
        f.append(Frequency * i / L)
    return f

def fft_frequency_domain(signal_data,Fs):
    L=len(signal_data)
    Y= fft(np.array(signal_data))
    P2 = round_list(abs(Y/L),4)
    P1 = mirror_signal_remove(P2)
    for i in range(1,len(P1)-1):
        P1[i] = 2*P1[i]
    f = round_list(frequency_domain_f(Fs,L),4)
    show_plot(f,P1,"f (Hz)","|P(f)|","Single-Sided Amplitude Spectrum","",[0,25])

def periodogram_frequency_domain(signal_data,Fs):
    L=len(signal_data)
    faxis, pxx = signal.periodogram(signal_data, Fs,window="hamming",nfft=L,scaling='density')
    show_plot(faxis, pxx, "f (Hz)", "PSD", "Periodogram","", [0, 25])

def welch_psd_estimation(signal,WINDOW,NOVERLAP,NFFT,Fs,onesided):
    faxis, pxx = scipy.signal.welch(signal, Fs, noverlap=NOVERLAP,nfft=NFFT,return_onesided=onesided,nperseg=WINDOW)
    show_plot(faxis, pxx, "f (Hz)", "PSD", "Welch-based PSD",str(NOVERLAP)+" samples-overlap", [0, 25])
time = time_axis(eeg,fs)
show_plot(time,eeg,'Time(S)','Signal Value','EEG Trace',"",[])
fft_frequency_domain(eeg,fs)
periodogram_frequency_domain(eeg,fs)
welch_psd_estimation(eeg,256,32,256,fs,True)
welch_psd_estimation(eeg,256,200,256,fs,True)