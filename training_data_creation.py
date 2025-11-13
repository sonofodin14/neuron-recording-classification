# Third-Party Imports
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np

# Load Waveform
mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']

# Split data into 250 wide blocks with steps of 160
windows = []
indexes  = []
window_length = 250
step = 200
stop_index = len(d) - window_length
current_index = 0
while current_index < stop_index:
    windows.append(d[current_index:current_index+window_length])
    indexes.append(list(range(current_index, current_index+window_length)))
    current_index += step

# Check if spike index is within the the first 200 samples
contains_spikes = [] # 0 if no, 1 if yes
search_set = set(Index)

for item in indexes:
    if not search_set.isdisjoint(item):
        contains_spikes.append(1)
    else:
        contains_spikes.append(0)


# plt.subplot(2,1,1)
plt.plot(indexes[8],windows[8])
plt.plot(indexes[7],windows[7], linestyle='dotted')
plt.plot(indexes[9],windows[9], linestyle='dotted')

plt.show()