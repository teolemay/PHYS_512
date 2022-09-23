import numpy as np
import matplotlib.pyplot as plt

#get previous fit values
cheb = np.loadtxt('cheb_log2_coeffs.txt')
#calculate log_2 of e once.
log2e = np.log2(np.e)

def mylog2(x, cheb=cheb, log2e=log2e):
    mantissa, exponent = np.frexp(x)
    mantFit = mantissa*4 - 3 #rescale mantissa values for chebyshev fit
    return (np.polynomial.chebyshev.chebval(mantFit, cheb) + exponent) / log2e


Xshow = np.linspace(0.01, 50000, 100000)

plt.figure()
plt.title(r'Natural logarithm using Chebyshev fit of $\log_2$')
plt.plot(Xshow, np.log(Xshow), label=r'$\ln(x)$')
plt.plot(Xshow, mylog2(Xshow), '-.', label='mylog2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


plt.figure()
plt.title(r'Absolute error of mylog2 function vs. $\ln(x)$')
plt.plot(Xshow, np.abs(np.log(Xshow) - mylog2(Xshow)))
plt.xlabel('x')
plt.ylabel('Error')
plt.show()
plt.show()

