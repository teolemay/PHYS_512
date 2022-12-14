import numpy as np
import matplotlib.pyplot as plt

data = np.load('sidebands.npz')
t = data['time']
d = data['signal']

#numerical differentiation
def numderiv(fun, x, args):
    dx = np.cbrt(1e-16)*x
    return ( fun( x+dx , args) - fun( x-dx , args) ) / (2*dx)

def triple_lorentzian(args):
    t = args[0]
    a = args[1]
    t0 = args[2]
    w = args[3]
    b = args[4]
    c = args[5]
    dt = args[6]
    one = a / ( 1 + ( (t - t0)**2 / ( w**2 ) ) ) 
    two = b / ( 1 + ( (t - t0 + dt)**2 / ( w**2 ) ) ) 
    three = c / ( 1 + ( (t - t0 - dt)**2 / ( w**2 ) ) ) 
    return one + two + three

#this function lets me choose a variable to differentiate with respect to (that's not t)
def triple_lorentzian_wrt(x, params):
    #x is the parameter we are differentiating around
    #params is all the args for lorentzian + index of variable for differentiation in list.
    # params = [t, a, t0, w, idx]
    idx = params[-1]
    args = params[:-1]
    args[idx] = x
    return triple_lorentzian(args)

# Lorentzian model with analytical derivatives
def newt_lorentzian_numerical(p, t):
    L = triple_lorentzian([t, p[0], p[1], p[2], p[3], p[4], p[5]])
    grad = np.zeros((t.size, p.size))
    #should do a for loop here, I chose not to for aesthetic reasons.
    #dLda
    grad[:, 0] = numderiv(triple_lorentzian_wrt, p[0], [t, p[0], p[1], p[2], p[3], p[4], p[5], 1])
    #dLdt0
    grad[:, 1] = numderiv(triple_lorentzian_wrt, p[1], [t, p[0], p[1], p[2], p[3], p[4], p[5], 2])
    #dLdw
    grad[:, 2] = numderiv(triple_lorentzian_wrt, p[2], [t, p[0], p[1], p[2], p[3], p[4], p[5], 3])
    #dLdb
    grad[:, 3] = numderiv(triple_lorentzian_wrt, p[3], [t, p[0], p[1], p[2], p[3], p[4], p[5], 4])
    #dLdc
    grad[:, 4] = numderiv(triple_lorentzian_wrt, p[4], [t, p[0], p[1], p[2], p[3], p[4], p[5], 5])
    #dLd(dt)
    grad[:, 5] = numderiv(triple_lorentzian_wrt, p[5], [t, p[0], p[1], p[2], p[3], p[4], p[5], 6])

    return L, grad

# calculate magnitude of parameter update to set threshold for Newton's method.
def dp_magnitude(p):
    psum = 0
    for i in p:
        psum += i*i
    return np.sqrt(psum)

#initial guess
p0 = np.array([1.5, 0.0002, 0.00002, 0.1, 0.2, 0.00005])
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
print('b', p0[3])
print('c', p0[4])
print('dt', p0[5])


#naive noise and error estimation
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
print('b error', errM[3])
print('c error', errM[4])
print('dt error', errM[5])


#here we do mcmc
def chisq(t, d, params):
    y = triple_lorentzian([t] + [p for p in params])
    return np.sum((d - y)**2)

#now we can do mcmc from here.
#code copied/adapted mcmc implementation in "PHYS512_modelling.pdf" slides
nsteps = 100000
nchains = 1

chains = [None]*nchains
for k in range(nchains):
    chain = np.zeros((nsteps, len(p0)))
    chain[0, :] = np.array([3, 0.0002, 0.00002, 0.5, 0.5, 0.0001])
    chi = chisq(t, d, chain[0, :]) #set initial chisq to be kinda bad.
    for j in range(1, nsteps):
        newparams = chain[j-1, :] + np.random.randn(len(errM))*errM
        chi_new = chisq(t, d, newparams)
        if chi_new < chi:
            chi = chi_new
            chain[j, :] = newparams
        else:
            accept = np.exp(-0.5*(chi_new - chi))
            if accept > np.random.rand():
                chain[j, :] = newparams
            else:
                chain[j, :] = chain[j-1, :]
        print(f'step {j}', end='\r')
    chains[k] = chain

print(" "*20, end='\r')

for val in np.mean(np.mean(chains, axis=1), axis=0):
    print(val)
print()
print('errors')
for er in np.mean(np.std(chains, axis=1), axis=0):
    print(er)

#alarm for long mcmc processes.
import winsound
duration = 1000  # milliseconds
freq = 500  # Hz
winsound.Beep(freq, duration)

l = ['a', 't0', 'w', 'b', 'c', 'dt']
plt.subplots(2, 3)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(l[i])
    plt.plot(chains[0][:, i])
plt.subplot_tool()
plt.show()

plt.figure()
plt.title(r"MCMC fit")
plt.plot(t, d, label='Data')
plt.plot(t, triple_lorentzian([t] + [p for p in np.mean(chains[0], axis=0)]), label='Model')
plt.xlabel('t')
plt.xticks([0, 0.0001, 0.0002, 0.0003, 0.0004])
plt.ylabel('d')
plt.legend()
plt.show()








