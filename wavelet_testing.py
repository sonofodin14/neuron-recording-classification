# Third-Party Imports
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from skimage.restoration import denoise_wavelet
import scipy.signal as sig
from random import randint
import keras
from sklearn import preprocessing

def load_data():
    files = ['D2.mat', 'D3.mat', 'D4.mat', 'D5.mat', 'D6.mat']
    data_entries = []
    for file in files:
        mat = spio.loadmat(file, spueeze_me=True)
        data_entries.append(mat['d'])
    return data_entries[0], data_entries[1], data_entries[2], data_entries[3], data_entries[4]

def create_bp_filter(numtaps, fl, fu, fs):
    filter_coefficients = sig.firwin(numtaps, [fl, fu], pass_zero=False, window='hamming', fs=fs)
    return filter_coefficients

def bp_filter_data(data, filter_coefficients, numtaps):
    filtered = sig.lfilter(filter_coefficients, 1.0, data)
    filtered = np.roll(filtered, -(int(numtaps/2)))
    return filtered

def wavelet_denoising(data):
    denoised = denoise_wavelet(
        data, 
        method='BayesShrink', 
        mode='soft', 
        wavelet_levels=7,
        wavelet='bior3',
        rescale_sigma='True'
        )
    return denoised

def noise_dependent_peak_detection(data):
    # Calculating the Dynamic Peak Threshold dependent on standard deviation of data - this changes with noise
    stand_devs = [abs(x)/0.6745 for x in data]
    threshold = 4*np.median(stand_devs)

    peaks, _ = sig.find_peaks(data, height=threshold, distance=10, prominence=0.125)
    return peaks

def peaks_to_spike_index(peaks):
    index = np.asarray([x - 12 for x in peaks])

# Load Waveform
mat = spio.loadmat('D2.mat', squeeze_me=True)
d = mat['d']

# Normalise data
scaler = preprocessing.RobustScaler()
d_shaped = d.reshape(-1,1)
d_norm = scaler.fit_transform(d_shaped)
d_to_use = d_norm.flatten()

# Setup Filter
numtaps = 101
fl, fu =  250, 2750
filter_coef = sig.firwin(numtaps, [fl, fu], pass_zero=False, window='hamming', fs=25000)

# Filter & Shift Signal back
d_filt = sig.lfilter(filter_coef, 1.0, d_to_use)
d_filt = np.roll(d_filt, -(int(numtaps/2)))

# Wavelet denoising
d_denoise = denoise_wavelet(
    d_filt, 
    method='BayesShrink', 
    mode='hard', 
    wavelet_levels=3,
    wavelet='bior1.3',
    rescale_sigma='True'
    )

# Calculating the Dynamic Peak Threshold
sd_array = [abs(x)/0.6745 for x in d_denoise]
dynamic_threshold = 4*np.median(sd_array)

# Find Peaks and Create Index List
peaks, _ = sig.find_peaks(d_denoise, height=dynamic_threshold, distance=10, prominence=0.125)
Index = np.asarray([x - 12 for x in peaks])

# NEXT: 
# - slicing up waveform based on indexes for classification (index:index+75) -> needs to be same as training
# - make everything functions/classes
# - dynamic denoising/filtering for each dataset SNR
# - 

# Extracting Spike Waveforms
spikes = []
for index in Index:
    spikes.append(d_denoise[index:index+50])

# Load model and classify spikes
loaded_model = keras.models.load_model("best_model.keras")
# Class = []
# for spike in spikes:
    # Class.append(loaded_model.predict(spike))

# Reshape spikes to fit model input
spikes = np.asarray(spikes)
spikes = spikes.reshape((spikes.shape[0], spikes.shape[1], 1))
test_input = np.expand_dims(spikes[2], axis=0)
Class = loaded_model.predict(spikes, verbose=2)

print(Class)


# Pack Index and Class into .mat file



# Plots

# Index and peaks shown on data
# plt_max = 5000
# plt.figure(dpi=100)
# plt.hlines([dynamic_threshold], linestyle=[':'], xmin=0, xmax=plt_max)
# plt.plot(d_denoise[0:plt_max])
# plt.plot(d[0:plt_max], linestyle='dotted')
# plt.plot(peaks, d_denoise[peaks.astype(int)], "x", color='g')
# plt.plot(Index, d_denoise[Index.astype(int)], "o", color='r')
# plt.show()

# Show individual spike waveform
# plt.figure(dpi=100)
# plt.plot(spikes[randint(0, len(spikes))])
# plt.show()