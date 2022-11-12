import numpy as np
import matplotlib.pyplot as plt

k=0.6 
x = np.arange(0, 1001)
y = np.sin(k*x)

fft = np.fft.fft(y)
re = np.real(fft)
im = np.imag(fft)

DFT = (1 - np.exp(2*np.pi*x)) / ()