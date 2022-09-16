import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

data = np.loadtxt('lakeshore.txt')

def lakeshore(V, data):
    #do some interpolation
    #get T(V)
    #sort data in ascending voltage order.
    data = data[::-1, :]
    cubicspline = CubicSpline(data[:, 1], data[:, 0], bc_type='natural')
    T = cubicspline(V)

    #Error calculation:
    #what are the points around V?
    V = np.atleast_1d(V)
    error = np.zeros(V.shape)
    for i, vi in enumerate(V):
        if vi in data[:, 1]:
            error[i] = 0    #no error if on a datapoint!
        else:
            tmp = np.abs(data[:, 1] - vi)
            idx = np.sort(np.argsort(tmp, )[:4]) #4 points around the interpolation point
            #calculate second derivatives using known first derivatives.
            dd1 = (data[idx[1], 2] - data[idx[0], 2])/(data[idx[1], 1]-data[idx[0], 1]) 
            dd2 = (data[idx[2], 2] - data[idx[1], 2])/(data[idx[2], 1]-data[idx[1], 1])
            dd3 = (data[idx[3], 2] - data[idx[2], 2])/(data[idx[3], 1]-data[idx[2], 1])
            #calculate third derivative
            ddd2 = (dd2 - dd1)/(data[idx[2], 1] - data[idx[1], 1])
            ddd3 = (dd3 - dd2)/(data[idx[3], 1] - data[idx[2], 1])
            #fourth derivative:
            d4 = (ddd3 - ddd2)/(data[idx[3], 1] - data[idx[2], 1]) #4th derivative is offset, but have no other.
            #calculate error according to cubic polynomial error formula with the only 4th derivative I have.
            p = np.product( ( data[np.ix_(idx), 1] - vi ) )
            error[i] = np.abs( (1/24) * d4 * p )
    return T, error



#sample for testing/showing:
interval = np.linspace(data[-1, 1]+0.001, data[1, 1]+0.001, 500)
T, error = lakeshore(interval, data)

plt.figure()
plt.title('Lakeshore 670 diode interpolation')
plt.plot(data[:, 1], data[:, 0], 'ko', markersize=3, label='Diode data')
plt.plot(interval, T, label='Cubic spline interpolation') 
plt.xlabel('V')
plt.ylabel('T')
plt.legend()
plt.show()

plt.figure()
plt.title('Lakeshore 670 diode estimated interpolation error')
plt.plot(interval, error, label='Estimated Error') 
plt.xlabel('V')
plt.ylabel(r'$\Delta T$')
plt.legend()
plt.show()

