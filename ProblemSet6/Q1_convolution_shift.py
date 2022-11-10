import numpy as np
import matplotlib.pyplot as plt


def conv_shift(arr, dx):
    """
    Shift an array arr by dx 
    like a worse version of np.roll!
    """
    dx = len(arr)//2 + int(dx) # need to set the impulse position to index from the middle of the array!
    arr = np.array(arr)
    shift = np.zeros(arr.shape)
    shift[dx] = 1
    longarr = np.concatenate((arr, arr))
    return np.convolve(longarr, shift, 'same')[len(arr):]


if __name__ == "__main__":

    x = np.linspace(-5, 5, 201)
    g = np.exp(- x**2/2)/(np.sqrt(2*np.pi))


    plt.figure()
    plt.title('Convolution shift example')
    plt.plot(g, label='Input array')
    plt.plot(conv_shift(g, len(g)//2), label='Shifted array')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()