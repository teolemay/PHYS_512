import numpy as np
import matplotlib.pyplot as plt

a = 0.1
N = 1024
x = np.arange(0, N)
y = np.sin(a*x)

w = y * (0.5 - (0.5*np.cos(2*np.pi*x/N)))

fft = np.fft.fft(y)
wfft = np.fft.fft(w)


plt.figure()
plt.title('Power spectrum')
plt.plot((fft * np.conj(fft))[:40], label='Raw sine wave')
plt.plot((wfft * np.conj(wfft))[:40], label='Windowed sine wave')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.legend()

plt.show()