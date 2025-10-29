import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.signal as spsig
from random import randint

mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
sorted_Index = sorted(Index)
Class = mat['Class']

d_short = d[1400:1500]

sample_index = randint(0, len(sorted_Index))
print("Sample Index: ", sample_index)
d_sample = d[Index[sample_index]-5:Index[sample_index]+75]

peaks, _ = spsig.find_peaks(d, height=1, width=4, threshold=0.001)

print(len(peaks))

print(len(Index))

plt.plot(d_sample)
# plt.plot(Index, d[Index.astype(int)], "x", color='r')
# plt.plot(peaks, d[peaks.astype(int)], "x", color='g')
plt.show()