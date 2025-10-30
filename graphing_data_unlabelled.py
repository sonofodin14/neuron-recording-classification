import matplotlib.pyplot as plt
import scipy.io as spio
from scipy import fft
import scipy.integrate as it
import scipy.signal as sig
import numpy as np

mat = spio.loadmat('D2.mat', squeeze_me=True)
d = mat['d']

numtaps = 43
fl, fu =  400/12500, 3000/12500

filter_coef = sig.firwin(numtaps, [fl, fu], pass_zero=False, window='hamming')

filtered_d = sig.lfilter(filter_coef, 1.0, d)
# shifted_back_d = np.roll(filtered_d, -17)

window_size = 7
moving_avg =  np.convolve(d, np.ones(window_size)/window_size, mode='same')

sd_array = [abs(x)/0.6745 for x in filtered_d]
dynamic_threshold = 5*np.median(sd_array)

d_integral = it.cumulative_trapezoid(filtered_d)
scaled_d_int = [x*0.01 for x in d_integral]
dv_scaled = np.gradient(scaled_d_int)
dv_scaled_again = np.gradient(dv_scaled)

plt.plot(d)
# plt.plot(moving_avg)
plt.plot(filtered_d)
# plt.plot(shifted_back_d)
plt.hlines([dynamic_threshold], linestyle=[':'], xmin=0, xmax=len(d))
# plt.plot(d_integral)
# plt.plot(scaled_d_int)
# plt.plot(dv_scaled)
# plt.plot(dv_scaled_again)

# fft_d = fft.fft(d)
# abs_fft  = [abs(x) for x in fft_d]

# plt.plot(abs_fft[0:int(0.5*len(abs_fft))])
# plt.xscale('log')

plt.show()