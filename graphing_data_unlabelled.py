import matplotlib.pyplot as plt
import scipy.io as spio

mat = spio.loadmat('D6.mat', squeeze_me=True)
d = mat['d']

plt.plot(d)

plt.show()