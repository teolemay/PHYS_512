import numpy as np
import matplotlib.pyplot as plt

#True sample points
Xvals = np.linspace(0.5, 1, 51, endpoint=True)
Yvals = np.log2(Xvals)

#fitting Chebyshev polynomial
order=25
chebXvals = Xvals*4 - 3
cheb = np.polynomial.chebyshev.chebfit(chebXvals, Yvals, order)
print(cheb)

#evaluating Chebyshev polynomial
orduse = 8
Xuse = np.linspace(-1, 1, 1001)
pred = np.polynomial.chebyshev.chebval(Xuse, cheb[:orduse])

#rescaling interval over which fit was evaluated
Xshow = (Xuse + 3)/4

plt.figure()
plt.title(r'$\log_2$ Chebyshev fit')
plt.plot(Xshow, np.log2(Xshow), label=r'$\log_2(x)$')
plt.plot(Xshow, pred, '-.', label='Chebyshev')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.figure()
plt.title('Absolute error of Chebyshev fit')
plt.plot(Xshow, np.abs(np.log2(Xshow) - pred))
plt.xlabel('x')
plt.ylabel('Error')
plt.show()