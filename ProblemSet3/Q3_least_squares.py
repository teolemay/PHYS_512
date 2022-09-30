import numpy as np
import matplotlib.pyplot as plt

x, y, z = np.loadtxt('dish_zenith.txt', unpack=True)

A = np.zeros((len(x), 4))
#fill in A
A[:, 0] = 1
A[:, 1] = x
A[:, 2] = y
A[:, 3] = x**2 + y**2

#Copied from polyfit_class.py
lhs = A.T@A
rhs = A.T@z
mfit = np.linalg.inv(lhs)@rhs

a = mfit[3]
x_0 = mfit[1]/(-2*a)
y_0 = mfit[2]/(-2*a)
z_0 = mfit[0] - a*x_0**2 - a*y_0**2
print('a', a)
print('x_0', x_0)
print('y_0', y_0)
print('z_0', z_0)

fig = plt.figure()
ax = plt.axes(fig, projection='3d')
ax.plot()s