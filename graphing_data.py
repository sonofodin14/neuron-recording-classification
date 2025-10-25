import matplotlib.pyplot as plt
import scipy.io as spio

mat = spio.loadmat('D1.mat', squeeze_me=True)
d = mat['d']
Index = mat['Index']
Class = mat['Class']

d = d[1400:1500]

print(min(Index))

plt.plot(d)
plt.show()