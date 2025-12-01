from utils import load_training_data, SPIKE_WIDTH
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

OVERLAP = 100
WINDOW_WIDTH = 250

D1, Index, Class = load_training_data()

def separate_spike_classes():
    # Format: spike[x] = [d data], [index], [class]
    spikes = []

    # Extract windows of spikes from index
    for i in range(len(Index)):
        spikes.append([D1[Index[i]:Index[i]+SPIKE_WIDTH].tolist(), int(Index[i]), int(Class[i])])

    # Create individual lists of spike classes to explore features
    class_1_spikes = []
    class_2_spikes = []
    class_3_spikes = []
    class_4_spikes = []
    class_5_spikes = []

    for spike in spikes:
        match spike[2]:
            case 1:
                class_1_spikes.append(spike[0])
            case 2:
                class_2_spikes.append(spike[0])
            case 3:
                class_3_spikes.append(spike[0])
            case 4:
                class_4_spikes.append(spike[0])
            case 5:
                class_5_spikes.append(spike[0])
            case _:
                pass
    return class_1_spikes, class_2_spikes, class_3_spikes, class_4_spikes, class_5_spikes

def average_spike_windows(spike_array):
    return np.mean(spike_array, axis=0)

def replace_list_section(list, section_start_index, section_to_insert):
    end_index = section_start_index + len(section_to_insert)
    list[section_start_index:end_index] = np.array(section_to_insert)

def list_to_overlapping_windows(list, window_length, overlap):
    windows = []
    num_windows = int(len(list) / (window_length-overlap))
    i_increase = (window_length-overlap)
    for i in range(num_windows):
        start = i*i_increase
        end = start + window_length
        if end <= len(list):
            windows.append(np.asarray(list[start:end]))
    return np.stack(windows) if windows else np.array([])

def overlapping_windows_to_list(windows, overlap):
    list = []
    for window in windows:
        list = np.concatenate((list, window))
        list = list[0:-overlap]
        last_overlap_data = window[-overlap:]
    list = np.concatenate((list, last_overlap_data))
    return list    

def add_noise_individual(data, noise_level):
    shape = data.shape
    noise = np.random.normal(0, noise_level, shape)
    noisy_data = data + noise
    return noisy_data

def add_noise_multiple(data_windows, noise_level):
    noised_windows = []
    for i in range(len(data_windows)):
        noised_windows.append(add_noise_individual(data_windows[i], noise_level))
    return np.asarray(noised_windows)

def concat_arrays(*args: ArrayLike):
    return np.concatenate(args)

class_1, class_2, class_3, class_4, class_5 = separate_spike_classes()

class_1_mean = average_spike_windows(class_1)
class_2_mean = average_spike_windows(class_2)
class_3_mean = average_spike_windows(class_3)
class_4_mean = average_spike_windows(class_4)
class_5_mean = average_spike_windows(class_5)

'''
# plt.subplot(231)
# plt.plot(class_1_mean)

# plt.subplot(232)
# plt.plot(class_2_mean)

# plt.subplot(233)
# plt.plot(class_3_mean)

# plt.subplot(234)
# plt.plot(class_4_mean)

# plt.subplot(235)
# plt.plot(class_5_mean)

# plt.show()
'''

zeros = np.zeros(len(D1))

recon_data = zeros.copy()

# Replace every spike with the class average
for i in range(len(Index)):
    match Class[i]:
        case 1:
            replace_list_section(recon_data, Index[i], class_1_mean)
        case 2:
            replace_list_section(recon_data, Index[i], class_2_mean)
        case 3:
            replace_list_section(recon_data, Index[i], class_3_mean)
        case 4:
            replace_list_section(recon_data, Index[i], class_4_mean)
        case 5:
            replace_list_section(recon_data, Index[i], class_5_mean)
        case _:
            pass

# Ensure recon_data is a flat array
recon_data = np.asarray(recon_data).flatten()

windows_clean = list_to_overlapping_windows(recon_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)

windows_n1 = add_noise_multiple(windows_clean, 0.5)
windows_n2 = add_noise_multiple(windows_clean, 1.0)
windows_n3 = add_noise_multiple(windows_clean, 1.5)
windows_n4 = add_noise_multiple(windows_clean, 2.0)
windows_n5 = add_noise_multiple(windows_clean, 2.5)
windows_n6 = add_noise_multiple(windows_clean, 3.0)
windows_n7 = add_noise_multiple(windows_clean, 3.5)
windows_n8 = add_noise_multiple(windows_clean, 4.0)
windows_n9 = add_noise_multiple(windows_clean, 4.5)
windows_n10 = add_noise_multiple(windows_clean, 5.0)

noisy_inputs = concat_arrays(
    windows_n1,
    windows_n2,
    windows_n3,
    windows_n4,
    windows_n5,
    windows_n6,
    windows_n7,
    windows_n8,
    windows_n9,
    windows_n10,
)

expected_outputs = concat_arrays(
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
)
