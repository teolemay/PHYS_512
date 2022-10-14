import numpy as np
import matplotlib.pyplot as plt

data = np.load('sidebands.npz')
t = data['time']
d = data['signal']

# Lorentzian model with analytical derivatives
def newt_lorentzian(p, t):
    L = p[0] / ( 1 + ( (t - p[1])**2 / ( p[2]**2 ) ) ) 
    grad = np.zeros((t.size, p.size))
    #dLda
    grad[:, 0] = 1 / ( 1 + ( (t - p[1])**2 / ( p[2]**2 ) ) ) 
    #dLdt0
    grad[:, 1] = ( 2*p[0]*(t - p[1]) / (p[2]**2) ) / ( 1 + ( (t - p[1])**2 / ( p[2]**2 ) ) )**2
    #dLdw
    grad[:, 2] = ( 2*p[0]*((t - p[1])**2) / (p[2]**3) ) / ( 1 + ( (t - p[1])**2 / ( p[2]**2 ) ) )**2
    return L, grad

# calculate magnitude of parameter update to set threshold for Newton's method.
def dp_magnitude(p):
    psum = 0
    for i in p:
        psum += i*i
    return np.sqrt(psum)

#initial guess
p0 = np.array([10, 0.0002, 0.0001])
mag = 1
thresh = 1e-10
n_iter = 0

while (mag > thresh) and (n_iter < 10000) :
    n_iter += 1
    pred, grad = newt_lorentzian(p0, t)
    lhs = grad.T @ grad
    diff = (d - pred).T
    rhs = grad.T @ diff
    dp = np.linalg.inv(lhs) @ rhs
    p0 = p0 + dp

    mag = dp_magnitude(dp)

print(f'Threshold passed after {n_iter} steps')
print(f'final parameters')
print(p0)

noise = np.sqrt(np.mean( (d - pred)**2 ))
# N is I * noise so N^-1 is I*(1/noise).
# parameter covariance matrix is dm^2 = (A'^T N^-1 A')^-1
# A' is last grad from the calculation!
rhs = (grad.T * (1/noise) ) @ grad
covM = np.linalg.inv(rhs)
errM = np.sqrt(np.diag(covM))
print('a error', errM[0])
print('t0 error', errM[1])
print('w error', errM[2])


plt.figure()
plt.title(r"Data with Gaussian filter ($\sigma = 20$)")
plt.plot(t, d, label='Data')
plt.plot(t, pred, label='Model')
plt.xlabel('t')
plt.xticks([0, 0.0001, 0.0002, 0.0003, 0.0004])
plt.ylabel('d')
plt.legend()

plt.figure()
plt.title('Difference of data and model histogram')
plt.hist(d-pred, 50, density=True)
plt.xlabel('Differences')
plt.ylabel('Normalized counts')
plt.show()






