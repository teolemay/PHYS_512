import numpy as np
import matplotlib.pyplot as plt
from Q1_convolution_shift import conv_shift 

def dft_correlation(arr1, arr2):
    karr1 = np.fft.rfft(arr1)
    karr2 = np.fft.rfft(arr2)
    return np.fft.irfft(karr1 * np.conj(karr2))[:]

def shift_correlation(arr, dx):
    arr1 = np.array(arr)
    arr2 = conv_shift(arr, dx)
    return dft_correlation(arr2, arr1)


if __name__ == "__main__":
    

    x = np.linspace(-5, 5, 201)
    g = np.exp(- x**2/2)/(np.sqrt(2*np.pi))

    plt.figure()
    plt.title('Autocorrelation of a Gaussian')
    plt.plot(dft_correlation(g, g)[:len(g)//2])
    plt.xlabel(r'$\tau$')
    plt.ylabel('Correlation')

    plt.figure()
    plt.title('Correlation of Gaussian and Gaussian shifted by 20')
    plt.plot(shift_correlation(g, 20)[:len(g)//2])
    plt.xlabel(r'$\tau$')
    plt.ylabel('Correlation')
    plt.show()


