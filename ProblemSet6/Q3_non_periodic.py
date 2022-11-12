import numpy as np
import matplotlib.pyplot as plt
from Q1_convolution_shift import conv_shift
from Q2_correlation_function import dft_correlation

def non_periodic_correlation(arr1, arr2):
    paddarr1 = np.concatenate((np.zeros(len(arr1)//2), arr1, np.zeros(len(arr1)//2)))
    paddarr2 = np.concatenate((np.zeros(len(arr2)//2), arr2, np.zeros(len(arr2)//2)))
    karr1 = np.fft.rfft(paddarr1)
    karr2 = np.fft.rfft(paddarr2)
    return np.fft.irfft(karr1 * np.conj(karr2))[:len(arr1)//2]


x = np.linspace(-5, 5, 201)
g = np.exp(- x**2/2)/(np.sqrt(2*np.pi))
gs = conv_shift(g, 60)


plt.figure()
plt.title('Correlation function outputs')
plt.plot(dft_correlation(g, gs)[:len(g)//2], '-', label='DFT correlation')
plt.plot(non_periodic_correlation(g, gs), '--', label='Padded DFT correlation')
plt.xlabel(r'$\tau$')
plt.ylabel('Correlation')
plt.ylim(0, 2.01)
plt.legend()

plt.figure()
plt.title('Non-periodic functions')
plt.plot(g)
plt.plot(gs)
plt.xlabel('x')
plt.ylabel('y')

#now convolve pre-padded functions
g = np.concatenate((np.zeros(100), g, np.zeros(100)))
gs = np.concatenate((np.zeros(100), gs, np.zeros(100)))

plt.figure()
plt.title('Correlation function outputs')
plt.plot(dft_correlation(g, gs)[:100], '-', label='DFT correlation')
plt.plot(non_periodic_correlation(g, gs)[:100], '--', label='Padded DFT correlation')
plt.xlabel(r'$\tau$')
plt.ylabel('Correlation')
plt.ylim(0, 2.01)
plt.legend()

plt.figure()
plt.title('Padded non-periodic functions')
plt.plot(g)
plt.plot(gs)
plt.xlabel('x')
plt.ylabel('y')
plt.show()