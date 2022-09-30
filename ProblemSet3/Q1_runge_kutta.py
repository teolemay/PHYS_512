import numpy as np
import matplotlib.pyplot as plt

#default RK4 algorithm for 1 step - taken from Computational Physics by Mark Newman (2013)
def rk4_step(fun, x, y, h):
    k1 = h*fun(y, x)
    k2 = h*fun(y+k1/2, x+h/2)
    k3 = h*fun(y+k2/2, x+h/2)
    k4 = h*fun(y+k3, x+h)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6

#modified RK4 with half steps
def rk4_stepd(fun, x, y, h):
    Hstep = rk4_step(fun, x, y, h)
    half1 = rk4_step(fun, x, y, h/2)
    half2 = rk4_step(fun, x+(h/2), half1, h/2)
    return (16*(half2) - Hstep ) / 15

#driver function - computes basic RK4 over interval with given initial value
def rk4_integrator(fun, x_interval, y_initial, Nsteps):
    xmin, xmax = x_interval
    xvals = np.linspace(xmin, xmax, Nsteps)
    h = xvals[1] - xvals[0]

    ypred = np.zeros(xvals.shape)
    y = y_initial
    for i, x in enumerate(xvals):
        ypred[i] = y
        y = rk4_step(fun, x, y, h)

    return xvals, ypred

#driver function - computes modified RK4 over interval with given initial value
def modified_rk4_integrator(fun, x_interval, y_initial, Nsteps):
    xmin, xmax = x_interval
    xvals = np.linspace(xmin, xmax, Nsteps)
    h = xvals[1] - xvals[0]

    ypred = np.zeros(xvals.shape)
    y = y_initial
    for i, x in enumerate(xvals):
        ypred[i] = y
        y = rk4_stepd(fun, x, y, h)

    return xvals, ypred

def f(y, x):
    return y/(1+x**2)


N = 201 #200 steps = 201 points.
xmin = -20
xmax = 20
xvalsh, ypredh = rk4_integrator(f, (xmin, xmax), 1, N)
xvalsh2, ypredh2 = modified_rk4_integrator(f, (xmin, xmax), 1, N//3)

#evaluate actual answer for the different number of steps.
truthXh = np.linspace(xmin, xmax, N)
truthYh = ( 1/(np.exp(np.arctan(-20))) ) * np.exp(np.arctan(truthXh))
rmsH = np.sqrt( np.sum( (truthYh - ypredh)**2 ) )

truthXh2 = np.linspace(xmin, xmax, N//3)
truthYh2 = ( 1/(np.exp(np.arctan(-20))) ) * np.exp(np.arctan(truthXh2))
rmsH2 = np.sqrt( np.sum( (truthYh2 - ypredh2)**2 ) )

#print error
print(f'One step RK4 with {N-1} steps RMS: {rmsH}')
print(f'Modified Rk4 with {N//3 -1} steps RMS: {rmsH2}')

#plot figure
plt.figure()
plt.title(r'RK4 integration of $dy/dx = \frac{y}{1 + x^2}$')
plt.plot(truthXh, truthYh, label=r'$c_0 \exp\{ \arctan{x} \}$')
plt.plot(xvalsh, ypredh, '.', color='tab:orange', markersize=5, label='One step RK4')
plt.plot(xvalsh2, ypredh2, 'r+',color='k', markersize=5, label='Modified RK4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()