import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 201)
g = np.exp(- x**2/2)/(np.sqrt(2*np.pi))


def conv_shift(arr, dx):
    """
    Shift an array arr by dx 
    like a worse version of np.roll!
    """
    dx = len(arr)//2 + int(dx)
    arr = np.array(arr)
    shift = np.zeros(arr.shape)
    shift[dx] = 1
    longarr = np.concatenate((arr, arr))
    return np.convolve(longarr, shift, 'same')[len(arr):]


plt.figure()
plt.title('Convolution shift example')
plt.plot(g, label='Input array')
plt.plot(conv_shift(g, len(g)//2), label='Shifted array')
plt.legend()
plt.show()