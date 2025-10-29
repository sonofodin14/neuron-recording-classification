# Third-party imports
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.signal as spsig
from scipy import fft
import numpy as np

# Load in D1 training data
mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
sorted_Index = sorted(Index)
Class = mat['Class']

# Format: spike[x] = [d data], [index], [class]
spikes = []

# Extract windows of spikes from index
for i in range(len(Index)):
    # spikes.append([d[Index[i]-5:Index[i]+75].tolist(), int(Index[i]), int(Class[i])])
    spikes.append([d[Index[i]:Index[i]+75].tolist(), int(Index[i]), int(Class[i])])
# print(spikes)

# for i in range(3):
#     plt.plot(spikes[i][0])

# Create individual lists of spike classes to explore features
class_1_spikes = []
class_2_spikes = []
class_3_spikes = []
class_4_spikes = []
class_5_spikes = []

for spike in spikes:
    match spike[2]:
        case 1:
            class_1_spikes.append(spike)
        case 2:
            class_2_spikes.append(spike)
        case 3:
            class_3_spikes.append(spike)
        case 4:
            class_4_spikes.append(spike)
        case 5:
            class_5_spikes.append(spike)
        case _:
            pass

fft_class_1 = []
for i in range(len(class_4_spikes)):
    fft_sample = fft.fft(class_4_spikes[i][0])
    fft_class_1.append(fft_sample[0:int(len(fft_sample)/2)])

for i in range(len(fft_class_1)):
    plt.plot(fft_class_1[i])

# for spike in class_5_spikes:
#     plt.plot(spike[0])

plt.show()