from utils import load_training_data, minmax_scale, SPIKE_WIDTH, create_hp_filter, filter_data
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

OVERLAP = 64
WINDOW_WIDTH = 256

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

def add_white_noise(data, noise_level):
    num_samples = len(data)
    noise = np.random.normal(0, noise_level, num_samples)
    noisy_data = data + noise
    return noisy_data

def add_brownian_noise(data, noise_level):
    num_samples = len(data)
    # Start with white noise as the base signal
    white_noise = np.random.normal(0, noise_level, num_samples)
    # Perform a cumulative sum (integration) to create the brownian effect
    # This accumulates previous values, creating the characteristic low-frequency emphasis
    brownian_noise = np.cumsum(white_noise)
    return data + brownian_noise

def add_brownian_white_noise(data, white_noise_level, brownian_noise_level=0.025):
    white_noised_data = add_white_noise(data, white_noise_level)
    white_brownian_data = add_brownian_noise(white_noised_data, brownian_noise_level)
    return white_brownian_data
# def add_brownian_multiple(data_windows, noise_level):
#     noised_windows = []
#     for i in range(len(data_windows)):
#         noised_windows.append(add_brownian_noise(data_windows[i], noise_level))
#     return np.asarray(noised_windows)

# def add_noise_multiple(data_windows, noise_level):
#     noised_windows = []
#     for i in range(len(data_windows)):
#         noised_windows.append(add_white_noise(data_windows[i], noise_level))
#     return np.asarray(noised_windows)

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

# for i in range(len(Index)):
#     replace_list_section(recon_data, Index[i], D1[Index[i]:Index[i]+SPIKE_WIDTH].tolist())

numtaps = 1501
fc = 50
fs = 25000
filter_coef = create_hp_filter(numtaps, fc, fs)

# Ensure recon_data is a flat array
perfect_data = np.asarray(recon_data).flatten()
n1_data = add_brownian_white_noise(perfect_data, 0.5)
n1_data = filter_data(n1_data, filter_coef, numtaps)

n2_data = add_brownian_white_noise(perfect_data, 1.0)
n2_data = filter_data(n2_data, filter_coef, numtaps)

n3_data = add_brownian_white_noise(perfect_data, 1.5)
n3_data = filter_data(n3_data, filter_coef, numtaps)

n4_data = add_brownian_white_noise(perfect_data, 2.0)
n4_data = filter_data(n4_data, filter_coef, numtaps)

n5_data = add_brownian_white_noise(perfect_data, 2.5)
n5_data = filter_data(n5_data, filter_coef, numtaps)

n6_data = add_brownian_white_noise(perfect_data, 3.0)
n6_data = filter_data(n6_data, filter_coef, numtaps)

n7_data = add_brownian_white_noise(perfect_data, 3.5, brownian_noise_level=0.05)
n7_data = filter_data(n7_data, filter_coef, numtaps)

n8_data = add_brownian_white_noise(perfect_data, 4.0, brownian_noise_level=0.05)
n8_data = filter_data(n8_data, filter_coef, numtaps)

n9_data = add_brownian_white_noise(perfect_data, 4.5, brownian_noise_level=0.05)
n9_data = filter_data(n9_data, filter_coef, numtaps)

n10_data = add_brownian_white_noise(perfect_data, 5.0, brownian_noise_level=0.05)
n10_data = filter_data(n10_data, filter_coef, numtaps)

n11_data = add_white_noise(perfect_data, 0.5)
n11_data = filter_data(n11_data, filter_coef, numtaps)

n12_data = add_white_noise(perfect_data, 1.0)
n12_data = filter_data(n12_data, filter_coef, numtaps)

n13_data = add_white_noise(perfect_data, 1.5)
n13_data = filter_data(n13_data, filter_coef, numtaps)

n14_data = add_white_noise(perfect_data, 2.0)
n14_data = filter_data(n14_data, filter_coef, numtaps)

n15_data = add_white_noise(perfect_data, 2.5)
n15_data = filter_data(n15_data, filter_coef, numtaps)

n16_data = add_white_noise(perfect_data, 4.0)
n16_data = filter_data(n16_data, filter_coef, numtaps)

# all_noisy_data = [n1_data, n2_data, n3_data, n4_data, n5_data, n6_data, n7_data, n8_data, n9_data, n10_data, n11_data, n12_data, n13_data, n14_data, n15_data, n16_data]

windows_clean = list_to_overlapping_windows(perfect_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n1 = list_to_overlapping_windows(n1_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n2 = list_to_overlapping_windows(n2_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n3 = list_to_overlapping_windows(n3_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n4 = list_to_overlapping_windows(n4_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n5 = list_to_overlapping_windows(n5_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n6 = list_to_overlapping_windows(n6_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n7 = list_to_overlapping_windows(n7_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n8 = list_to_overlapping_windows(n8_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n9 = list_to_overlapping_windows(n9_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n10 = list_to_overlapping_windows(n10_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n11 = list_to_overlapping_windows(n11_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n12 = list_to_overlapping_windows(n12_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n13 = list_to_overlapping_windows(n13_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n14 = list_to_overlapping_windows(n14_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n15 = list_to_overlapping_windows(n15_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)
windows_n16 = list_to_overlapping_windows(n16_data, window_length=WINDOW_WIDTH, overlap=OVERLAP)

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
    windows_n11,
    windows_n12,
    windows_n13,
    windows_n14,
    windows_n15,
    windows_n16,
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
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
    windows_clean,
)

# Shuffle the combined data
idx = np.random.permutation(len(noisy_inputs))
noisy_inputs = noisy_inputs[idx]
expected_outputs = expected_outputs[idx]