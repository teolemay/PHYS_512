import numpy as np
import matplotlib.pyplot as plt

a = 0.1
N = 1024
x = np.arange(0, N)
y = np.sin(a*x)

fft = np.fft.fft(y)
re = np.real(fft)
im = np.imag(fft)

def analytical_dft(k, N, a):
    lhs = (1 - (np.cos(a*N) + 1j*np.sin(a*N)) ) / (1 - (np.cos((-2*np.pi*k/N) + a) + 1j*np.sin((-2*np.pi*k/N) + a)))
    rhs = (1 - (np.cos(a*N) - 1j*np.sin(a*N)) ) / (1 - (np.cos((2*np.pi*k/N) + a) - 1j*np.sin((2*np.pi*k/N) + a)))
    return (1/2j) * (lhs - rhs)

DFT = analytical_dft(x, N, a)

plt.figure()
plt.title('Real')
plt.plot(np.abs(re - np.real(DFT)), label='difference')
plt.ylabel('|FFT - DFT|')
plt.xlabel('k')
plt.legend()

plt.figure()
plt.title('Imaginary')
plt.plot(np.abs(im - np.imag(DFT)), label='difference')
plt.xlabel('k')
plt.ylabel('|FFT - DFT|')
plt.legend()

plt.figure()
plt.title('Power spectrum')
plt.plot(DFT * np.conj(DFT), '.')
plt.xlabel('k')
plt.ylabel('Amplitude')

plt.show()


