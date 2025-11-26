from utils import load_training_data, 
import matplotlib.pyplot as plt
import numpy as np



D1, Index, Class = load_training_data()

def separate_spike_classes():
    # Format: spike[x] = [d data], [index], [class]
    spikes = []

    # Extract windows of spikes from index
    for i in range(len(Index)):
        # spikes.append([d[Index[i]-5:Index[i]+75].tolist(), int(Index[i]), int(Class[i])])
        spikes.append([D1[Index[i]:Index[i]+100].tolist(), int(Index[i]), int(Class[i])])

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

class_1, class_2, class_3, class_4, class_5 = separate_spike_classes()

class_1_mean = average_spike_windows(class_1)
class_2_mean = average_spike_windows(class_2)
class_3_mean = average_spike_windows(class_3)
class_4_mean = average_spike_windows(class_4)
class_5_mean = average_spike_windows(class_5)

plt.subplot(151)
plt.plot(class_1_mean)

plt.subplot(152)
plt.plot(class_2_mean)

plt.subplot(153)
plt.plot(class_3_mean)

plt.subplot(154)
plt.plot(class_4_mean)

plt.subplot(155)
plt.plot(class_5_mean)

plt.show()

