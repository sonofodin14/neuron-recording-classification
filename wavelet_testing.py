import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
from skimage.restoration import denoise_wavelet
import scipy.signal as sig

# Load Waveform
mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']

# Setup Filter
numtaps = 101
fl, fu =  250, 2750
filter_coef = sig.firwin(numtaps, [fl, fu], pass_zero=False, window='hamming', fs=25000)

# Filter & Shift Signal back
d_filt = sig.lfilter(filter_coef, 1.0, d)
d_filt = np.roll(d_filt, -(int(numtaps/2)))

# Wavelet denoising
d_denoise = denoise_wavelet(
    d_filt, 
    method='BayesShrink', 
    mode='soft', 
    wavelet_levels=7,
    wavelet='sym8',
    rescale_sigma='True'
    )

# Calculating the Dynamic Peak Threshold
sd_array = [abs(x)/0.6745 for x in d_denoise]
dynamic_threshold = 4*np.median(sd_array)

# Find Peaks and Create Index List
peaks, _ = sig.find_peaks(d_denoise, height=dynamic_threshold, distance=10, prominence=0.125)
indexes = np.asarray([x - 12 for x in peaks])

# Check D1 labels against calculated
# print(len(indexes), len(Index))

# Plots
plt.figure(dpi=100)
plt.hlines([dynamic_threshold], linestyle=[':'], xmin=0, xmax=len(d))
plt.plot(d_denoise)
plt.plot(peaks, d_denoise[peaks.astype(int)], "x", color='g')
plt.plot(Index, d_denoise[Index.astype(int)], "x", color='r')
plt.plot(indexes, d_denoise[indexes.astype(int)], "o", color='r')
plt.legend()
plt.show()