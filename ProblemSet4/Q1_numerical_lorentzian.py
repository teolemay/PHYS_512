import numpy as np
import matplotlib.pyplot as plt

data = np.load('sidebands.npz')
t = data['time']
d = data['signal']

#numerical differentiation
def numderiv(fun, x, args):
    dx = np.cbrt(1e-16)*x
    return ( fun( x+dx , args) - fun( x-dx , args) ) / (2*dx)

def lorentzian(args):
    t = args[0]
    a = args[1]
    t0 = args[2]
    w = args[3]
    return a / ( 1 + ( (t - t0)**2 / ( w**2 ) ) ) 

#this function lets me choose a variable to differentiate with respect to (that's not t)
def lorentzian_wrt(x, params):
    #x is the parameter we are differentiating around
    #params is all the args for lorentzian + index of variable for differentiation in list.
    # params = [t, a, t0, w, idx]
    idx = params[-1]
    args = params[:-1]
    args[idx] = x
    return lorentzian(args)

# Lorentzian model with analytical derivatives
def newt_lorentzian_numerical(p, t):
    L = lorentzian([t, p[0], p[1], p[2]])
    grad = np.zeros((t.size, p.size))
    #dLda
    grad[:, 0] = numderiv(lorentzian_wrt, p[0], [t, p[0], p[1], p[2], 1])
    #dLdt0
    grad[:, 1] = numderiv(lorentzian_wrt, p[1], [t, p[0], p[1], p[2], 2])
    #dLdw
    grad[:, 2] = numderiv(lorentzian_wrt, p[2], [t, p[0], p[1], p[2], 3])
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
    pred, grad = newt_lorentzian_numerical(p0, t)
    lhs = grad.T @ grad
    diff = (d - pred).T
    rhs = grad.T @ diff
    dp = np.linalg.inv(lhs) @ rhs
    p0 = p0 + dp

    mag = dp_magnitude(dp)

print(f'Threshold passed after {n_iter} steps')
print(f'final parameters')
print('a', p0[0])
print('t0', p0[1])
print('w', p0[2])

for p in p0:
    print(p)

plt.figure()
plt.title(r"Numerical derivative fit")
plt.plot(t, d, label='Data')
plt.plot(t, pred, label='Model')
plt.xlabel('t')
plt.xticks([0, 0.0001, 0.0002, 0.0003, 0.0004])
plt.ylabel('d')
plt.legend()
plt.show()







