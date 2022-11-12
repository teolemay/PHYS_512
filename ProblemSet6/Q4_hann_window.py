import numpy as np
import matplotlib.pyplot as plt

a = 0.1
N = 1024
x = np.arange(0, N)
y = np.sin(a*x)

w = y * (0.5 - (0.5*np.cos(2*np.pi*x/N)))

fft = np.fft.fft(y)
wfft = np.fft.fft(w)

newfft = np.zeros(fft.shape)
for i in range(1, len(newfft)-1):
    newfft[i] = fft[i] - (1/(N*4))*fft[i-1] - (1/(N*4))*fft[i+1]
newfft[0] = fft[0] - (1/(N*4))*fft[-1] - (1/(N*4))*fft[1]
newfft[-1] = fft[-1] - (1/(N*4))*fft[-2] - (1/(N*4))*fft[0]


plt.figure()
plt.title('Power spectrum')
plt.plot((fft * np.conj(fft))[:40], label='Raw sine wave')
plt.plot((wfft * np.conj(wfft))[:40], label='Windowed sine wave')
plt.plot((newfft * np.conj(newfft))[:40])
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.legend()

plt.show()