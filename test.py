import numpy as np
import matplotlib.pyplot as plt

text = 'hello world'*2

x = np.arange(0, 100)
y = np.sin(x/10)

plt.figure()
plt.plot(x, y, 'r+')
plt.text(40, 0, text.capitalize())
plt.show()