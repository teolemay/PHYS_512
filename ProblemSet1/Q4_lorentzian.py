import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def lorentzian(x):
    return 1./(1.+x**2)

fun = lorentzian

# evaluate function over 11 points to use for the interpolations
xsample = np.linspace(-1, 1, 8)
ysample = fun(xsample)
#x array to plot the interpolations, truth
xeval = np.linspace(xsample[0]+1e-15, xsample[-1], 500)
ytruth = fun(xeval)

#polynomial interpolation (cubic polynomial over each interval)
# this part uses code copied and adapted from the lecture slides for the cubic polynomial (interpolate_integrate.pdf)
poly = np.polyfit(xsample, ysample, 4)
ypoly = np.polyval(poly, xeval) #evaluates the polynomial using coefficients poly

plt.figure()
plt.title('Fourth order polynomial interpolation')
plt.plot(xeval, ytruth, label = r'$\frac{1}{1+x^2}$')
plt.plot(xeval, ypoly, '--', label='interpolation')
txt = 'Mean error: ' + "{:e}".format(np.mean(np.abs(ytruth-ypoly)))
plt.text(-0.5, 0.6, txt)
plt.ylabel(r'$y$')
plt.xlabel(r'$y$')
plt.legend()

#cubic spline interpolation
spline = interpolate.CubicSpline(xsample, ysample)
yspline = spline(xeval)

plt.figure()
plt.title('Cubic spline interpolation')
plt.plot(xeval, ytruth, label=r'$\frac{1}{1+x^2}$')
plt.plot(xeval, yspline, '--', label='interpolation')
txt = 'Mean error: ' + "{:e}".format(np.mean(np.abs(ytruth-yspline)))
plt.text(-0.5, 0.6, txt)
plt.ylabel(r'$y$')
plt.xlabel(r'$y$')
plt.legend()

#rational function interpolation
# this part uses code directly copied from the lecture slides (interpolate_integrate.pdf)
def rat_eval(p, q, x):
    top=0
    for i in range(len(p)):
        top = top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot = bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x, y, n, m):
    assert(len(x) == n+m-1)
    assert(len(y) == len(x))
    mat = np.zeros((n+m-1, n+m-1))
    for i in range(n):
        mat[:, i] = x**i
    for i in range(1, m):
        mat[:, i-1+n]=-y*x**i
    pars = np.dot(np.linalg.pinv(mat), y)
    p = pars[:n]
    q = pars[n:]
    return p, q

n = 4
m = 5
p, q = rat_fit(xsample, ysample, n, m)
print('p', p)
print('q', q)
yrat = rat_eval(p, q, xeval)

plt.figure()
plt.title('Higher order rational function interpolation (pinv)')
plt.plot(xeval, ytruth, label=r'$\frac{1}{1+x^2}$')
plt.plot(xeval, yrat, '--', label='interpolation')
txt = 'Mean error: ' + "{:e}".format(np.mean(np.abs(ytruth-yrat)))
plt.text(-0.5, 0.6, txt)
plt.ylabel(r'$y$')
plt.xlabel(r'$y$')
plt.legend()
plt.show()
