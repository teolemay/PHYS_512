import numpy as np
import matplotlib.pyplot as plt

x, y, z = np.loadtxt('dish_zenith.txt', unpack=True)

A = np.zeros((len(x), 4), )
#fill in A
A[:, 0] = 1
A[:, 1] = x
A[:, 2] = y
A[:, 3] = x**2 + y**2

#Copied from polyfit_class.py
lhs = A.T@A
rhs = A.T@z
mfit = np.linalg.inv(lhs)@rhs

#extract model parameters from best fit parameters
a = mfit[3]
x_0 = mfit[1]/(-2*a)
y_0 = mfit[2]/(-2*a)
z_0 = mfit[0] - a*x_0**2 - a*y_0**2
print('a', a)
print('x_0', x_0)
print('y_0', y_0)
print('z_0', z_0)

#plot surface using parameters
xplot = np.linspace(np.min(x), np.max(x), 300)
yplot = np.linspace(np.min(y), np.max(y), 300)
xplot, yplot = np.meshgrid(xplot, yplot)
zplot = a*( (xplot-x_0)**2 + (yplot-y_0)**2 ) + z_0

fig = plt.figure( )
ax = fig.add_subplot(projection='3d')
dots = ax.scatter(x, y, z, color='r', marker='.', label='data')
surface = ax.plot_surface(xplot, yplot, zplot, color='tab:blue', alpha=0.4, label='Linear least squares fit')
surface._facecolors2d = surface._facecolor3d
surface._edgecolors2d = surface._edgecolor3d
ax.legend()
ax.set(xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)', title='Target positions and best fit surface')

#use a poor approximation for noise = data - model (assuming model is much better than data as if data very noisy)
noise = (z - (a*( (x-x_0)**2 + (y-y_0)**2 ) + z_0))

N = np.outer(noise, noise)

sigma= np.diag( np.linalg.inv( A.T@np.linalg.inv(N)@A ))
print('a error', np.sqrt(sigma[3]))




##residual plot, not needed.
# fig = plt.figure( )
# ax = fig.add_subplot(projection='3d')
# surface = ax.scatter(x, y, noise, color='black', marker='.')
# ax.set(xlabel='x (mm)', ylabel='y (mm)', zlabel='data - model (mm)', title='Dish data fit residual plot')
# plt.show()

