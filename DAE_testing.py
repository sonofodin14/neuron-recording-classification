# Third-Party Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# First-Party Imports
import utils
import DAE_funcs
from DAE_funcs import WINDOW_WIDTH, OVERLAP

noisy_data = utils.load_file_data("TESTING DATA/D6.mat")

# High-pass filter data
numtaps = 1501
fc = 50
fs = 25000
filter_coef = utils.create_hp_filter(numtaps, fc, fs)
filtered_data = utils.filter_data(noisy_data, filter_coef, numtaps)

noisy_windows = DAE_funcs.list_to_overlapping_windows(filtered_data, WINDOW_WIDTH, OVERLAP)

denoiser = utils.load_denoising_model()
predictions = denoiser.predict(noisy_windows, verbose=2)
clean_windows = np.squeeze(predictions, axis=-1)

clean_data = DAE_funcs.overlapping_windows_to_list(clean_windows, OVERLAP)

scaler = MinMaxScaler()
clean_data_shaped = np.asarray(clean_data).reshape(-1, 1)
clean_data_scaled = scaler.fit_transform(clean_data_shaped)

stand_devs = [abs(x)/0.6745 for x in clean_data_scaled]
threshold = 0.75*np.median(stand_devs)

plt.plot(clean_data_scaled)
plt.hlines([threshold], linestyle=[':'], xmin=0, xmax=len(clean_data_scaled))
plt.plot(noisy_data, linestyle="dotted")
plt.show()