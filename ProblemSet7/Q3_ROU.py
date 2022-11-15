import numpy as np
import matplotlib.pyplot as plt

def expdist(x):
    if x < 0:
        return 0
    else:
        return np.exp(-x)

def accept(u, v, f):
    if (u <= np.sqrt(f(v/u))):
        return True
    else:
        return False

def exp_ROU(N):
    output = np.zeros(N)
    total = 0
    for i in range(N):
        Next = False
        while not Next:
            total += 1
            u = np.random.rand()
            v = np.random.rand() * (2/np.e)
            if accept(u, v, expdist):
                Next = True
                output[i] = (v/u)
    return output, total

N = 100000
rand, total = exp_ROU(N)
print(f'Efficiency: {N/total}')


x = np.linspace(0, 12, 10000)
y = np.exp(-x)

plt.figure()
plt.title('ROU sampling exponential distribution')
plt.hist(rand, 100, density=True, label='ROU sample histogram')
plt.plot(x, y, label=r'$e^{-x}$')
plt.legend()
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.show()

        


    