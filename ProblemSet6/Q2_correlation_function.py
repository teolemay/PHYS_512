import numpy as np
import matplotlib.pyplot as plt

def dft_correlation(arr1, arr2):
    karr1 = np.fft.fft(arr1)
    karr2 = np.fft.fft(arr2)
    return np.fft.ifft(karr1 * np.conj(karr2))[:len(arr1)//2]


if __name__ == "__main__":
    x = np.linspace(-5, 5, 201)
    g = np.exp(- x**2/2)/(np.sqrt(2*np.pi))

    plt.figure()
    plt.title('Autocorrelation of a Gaussian')
    plt.plot(dft_correlation(g, g))
    plt.xlabel(r'$\tau$')
    plt.ylabel('Correlation')
    plt.show()