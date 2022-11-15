import numpy as np
import matplotlib.pyplot as plt

generator = np.random.default_rng(1234)


#change this variable to set the path to rand_points.txt if needed.
txt_location = ''

if len(txt_location) > 0:
    rand_points = np.loadtxt(txt_location + '\\' + 'rand_points.txt')
else:
    rand_points = np.loadtxt('rand_points.txt')

x = rand_points[:, 0]
y = rand_points[:, 1]
z = rand_points[:, 2]
print(len(x))

n=100000000
py_rand = generator.integers(0, 2**31, size=(n, 3))
allow = np.all(py_rand < 1e8, axis=1)
pyx = py_rand[allow, 0]
pyy = py_rand[allow, 1]
pyz = py_rand[allow, 2]
print(len(pyx))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# # switching the commented line below changes the plot from the samples from the C PRNG to numpy PRNG.
ax.scatter(x, y, z, s=1)      
# ax.scatter(pyx, pyy, pyz, s=1)
plt.show()

