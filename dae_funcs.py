from utils import load_training_data, minmax_scale, SPIKE_WIDTH
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

OVERLAP = 125
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

def add_brownian_noise(data, noise_level):
    num_samples = len(data)
    # Start with white noise as the base signal
    white_noise = np.random.normal(0, noise_level, num_samples)
    # Perform a cumulative sum (integration) to create the brownian effect
    # This accumulates previous values, creating the characteristic low-frequency emphasis
    brownian_noise = np.cumsum(white_noise)
    # Normalize to prevent clipping (values going beyond -1 to 1 range)
    # brownian_noise = brownian_noise / np.max(np.abs(brownian_noise))
    return data + brownian_noise

def add_brownian_multiple(data_windows, noise_level):
    noised_windows = []
    for i in range(len(data_windows)):
        noised_windows.append(add_brownian_noise(data_windows[i], noise_level))
    return np.asarray(noised_windows)

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

# for i in range(len(Index)):
#     replace_list_section(recon_data, Index[i], D1[Index[i]:Index[i]+SPIKE_WIDTH].tolist())

# Ensure recon_data is a flat array
perfect_data = np.asarray(recon_data).flatten()
n1_data = add_brownian_noise(perfect_data, 0.5)
n2_data = add_brownian_noise(perfect_data, 1.0)
n3_data = add_brownian_noise(perfect_data, 1.5)
n4_data = add_brownian_noise(perfect_data, 2.0)
n5_data = add_brownian_noise(perfect_data, 2.5)
n6_data = add_brownian_noise(perfect_data, 3.0)
n7_data = add_brownian_noise(perfect_data, 3.5)
n8_data = add_brownian_noise(perfect_data, 4.0)
n9_data = add_brownian_noise(perfect_data, 4.5)
n10_data = add_brownian_noise(perfect_data, 5.0)
n11_data = add_brownian_noise(perfect_data, 5.5)
n12_data = add_brownian_noise(perfect_data, 6.0)
n13_data = add_brownian_noise(perfect_data, 6.5)
n14_data = add_brownian_noise(perfect_data, 7.0)
n15_data = add_brownian_noise(perfect_data, 7.5)
n16_data = add_brownian_noise(perfect_data, 8.0)

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

# windows_n1 = add_brownian_multiple(windows_clean, 0.5)
# windows_n2 = add_brownian_multiple(windows_clean, 1.0) 
# windows_n3 = add_brownian_multiple(windows_clean, 1.5)
# windows_n4 = add_brownian_multiple(windows_clean, 2.0)
# windows_n5 = add_brownian_multiple(windows_clean, 2.5)
# windows_n6 = add_brownian_multiple(windows_clean, 3.0)
# windows_n7 = add_brownian_multiple(windows_clean, 3.5)
# windows_n8 = add_brownian_multiple(windows_clean, 4.0)
# windows_n9 = add_brownian_multiple(windows_clean, 4.5)
# windows_n10 = add_brownian_multiple(windows_clean, 5.0)
# windows_n11 = add_brownian_multiple(windows_clean, 5.5)
# windows_n12 = add_brownian_multiple(windows_clean, 6.0)
# windows_n13 = add_brownian_multiple(windows_clean, 6.5)
# windows_n14 = add_brownian_multiple(windows_clean, 7.0)
# windows_n15 = add_brownian_multiple(windows_clean, 7.5)
# windows_n16 = add_brownian_multiple(windows_clean, 8.0)

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