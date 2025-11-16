import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.signal as spsig
from random import randint

mat = spio.loadmat('D2.mat', squeeze_me=True)
d = mat['d']

b, a = spsig.butter(5, [300, 3000], fs=25000, btype='band')
d_filtered = spsig.lfilter(b, a, d)

plt.plot(d, color='r')
plt.plot(d_filtered, color='g')
plt.show()