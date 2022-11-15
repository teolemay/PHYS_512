import numpy as np
import matplotlib.pyplot as plt

def expdist(x):
    if x >= 0:
        return np.exp(-x)
    return 0

def cauchydist(x):
    return 1/(1 + x**2)

def invCauchy(x):
    return np.tan(np.pi*x - (np.pi/2))

def rejection_sample(f, g, gdist, N):
    """
    rejection sampling of a distribution f from a bounding distribution g.
    f: target distribution probability density function
    g: sampling distribution probability density function
    gdist: sampling distribution inverse CDF for producing random samples from uniform dist on [0, 1]
    N: length of output.
    """
    output = np.zeros(N)
    total = 0
    for i in range(N):
        Next = False
        while not Next:
            total += 1
            xg = gdist(np.random.rand())
            u = np.random.rand()
            if u < ( f(xg)/g(xg) ):
                output[i] = xg
                Next = True

    return output, total

N = 100000
expsample, totaltries = rejection_sample(expdist, cauchydist, invCauchy, N)
print(f'Efficiency: {N/totaltries}')

plt.figure()
plt.title('Rejection sampling exponential distribution')
plt.hist(expsample, 100, density=True, label='Normalized rejection sampling histogram')
plt.plot(np.linspace(0, 10, 1000), np.exp(-np.linspace(0, 10, 1000)), label=r'$e^{-x}$')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.show()

