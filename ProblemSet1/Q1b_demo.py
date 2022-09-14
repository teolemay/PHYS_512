import numpy as np
import matplotlib.pyplot as plt

def func1(x):
    return np.exp(x)

def func2(x):
    return np.exp(0.01*x)

def deriv(func, x, dx):
    return (8*func(x+dx)-8*func(x-dx) + func(x-2*dx) - func(x+2*dx)) / (12*dx)


interval = np.linspace(-2, 2, 20)
true1 = func1(interval)
myderiv1 = deriv(func1, interval, 9.441e-4)
bigderiv1 = deriv(func1, interval, 9.441e-3)
smallderiv1 = deriv(func1, interval, 9.44e-5)

plt.figure()
plt.plot(interval, (true1-myderiv1), label=r'$\delta = 9.441e-4$')
plt.plot(interval, (true1-smallderiv1), label=r'$\delta/10$')
plt.title(r'Four point derivative error ($f(x)=e^x$)')
plt.xlabel(r'$x$')
plt.ylabel(r'Error')
plt.legend()

plt.figure()
plt.plot(interval, (true1-myderiv1), label=r'$\delta = 9.441e-4$')
plt.plot(interval, (true1-bigderiv1), label=r'$\delta \cdot 10$')
plt.title(r'Four point derivative error ($f(x)=e^x$)')
plt.xlabel(r'$x$')
plt.ylabel(r'Error')
plt.legend()

interval2 = np.linspace(-200, 200, 20)
true2 = 0.01*func2(interval2) #derivative of exp(0.01x) is 0.01*exp(0.01x)
myderiv2 = deriv(func2, interval2, 9.441e-2)
bigderiv2 = deriv(func2, interval2, 9.441e-1)
smallderiv2 = deriv(func2, interval2, 9.441e-3)

plt.figure()
plt.plot(interval2, (true2-myderiv2), label=r'$\delta = 9.441e-2$')
plt.plot(interval2, (true2-smallderiv2), label=r'$\delta/10$')
plt.title(r'Four point derivative error ($f(x)=e^{0.01x}$)')
plt.xlabel(r'$x$')
plt.ylabel(r'Error')
plt.legend()

plt.figure()
plt.plot(interval2, (true2-myderiv2), label=r'$\delta = 9.441e-2$')
plt.plot(interval2, (true2-bigderiv2), label=r'$\delta \cdot 10$')
plt.title(r'Four point derivative error ($f(x)=e^{0.01x}$)')
plt.xlabel(r'$x$')
plt.ylabel(r'Error')
plt.legend()
plt.show()
