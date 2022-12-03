import numpy as np
import matplotlib.pyplot as plt

pp = np.load('periodic_potential.npy')
pk = np.load('periodic_kinetic.npy')
npp = np.load('nonperiodic_potential.npy')
npk = np.load('nonperiodic_kinetic.npy')

t = np.arange(len(pp))*0.01

plt.figure()
plt.plot(t, pp, label='Gravitational potential energy')
plt.plot(t, pk*1/2, label='Kinetic energy')
plt.legend()
plt.xlabel('t')
plt.ylabel('Scaled energy (arb. units)')

plt.figure()
plt.title('Total energy over time')
plt.plot(t, pk-(0.5*pp))
plt.xlabel('t')
plt.ylabel('Scaled energy (arb. units)')


plt.show()