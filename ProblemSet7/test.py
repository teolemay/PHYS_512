import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 10000)
y = np.exp(-x)


xx = np.random.rand(10000)
yy = np.log(xx)

plt.figure()
plt.plot(x, y)
plt.plot(xx, yy, '.')
plt.show()