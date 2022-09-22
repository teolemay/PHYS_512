import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from Q2_adaptive_integration import adaptive_integrator

#will not use these in order to get rid of messy units
# k = 1e-8 # approximately 1/(4*eps_0)
# rho = 1 #surface charge density

R = 1
zvals = np.linspace(0, 8, 1001)

Equad = np.empty(zvals.shape, dtype=np.float64) 
Eadaptive = np.ones(zvals.shape, dtype=np.float64) 

for i, z in enumerate(zvals):

    def fun(theta):
        numerator = 2*np.pi*R*np.sin(theta)*(z + R*np.cos(theta))
        denominator = ( ( z + R*np.cos(theta) )**2 + ( R*np.sin(theta) )**2 )**(3/2)
        return numerator/denominator

    Equad[i] = quad(fun, 0, np.pi)[0] #quad returns: (integral value, estimated absolute error)
    Eadaptive[i] = adaptive_integrator(fun, 0, np.pi)

plt.figure()
plt.plot(zvals, Equad)
plt.title('Scipy.integrate.quad')
plt.xlabel('z')
plt.ylabel(r'$\frac{E}{k\rho}$')

plt.figure()
plt.plot(zvals, Eadaptive)
plt.title('Adaptive step size integrator')
plt.xlabel('z')
plt.ylabel(r'$\frac{E}{k\rho}$')

plt.figure()
plt.plot(zvals, np.abs(Equad - Eadaptive))
plt.title('Difference between integrals')
plt.xlabel('z')
plt.ylabel('Difference (absolute value)')

plt.show()



