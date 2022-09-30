import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#half-lives converted to seconds
half_lives = [
    4.46e9*365*24*60*60,
    24.1*24*60*60,
    6.70*60*60,
    245500*365*24*60*60,
    75380*365*24*60*60,
    1600*365*24*60*60,
    3.8235*24*60*60,
    3.1*60,
    26.8*60,
    19.9*60,
    164.3e-6,
    22.3*365*24*60*60,
    5.015*365*24*60*60,
    138.376*24*60*60
]

labels = [
    'U238',
    'Th234',
    'Pa234',
    'U234',
    'Th230',
    'Ra226',
    'Rn222',
    'Po218',
    'Pb214',
    'Bi214',
    'Po214',
    'Pb210',
    'Bi210',
    'Po210',
    'Pb206'
]


def decay_fun(x, y, hlives=half_lives):
    ln2 = np.log(2)
    dydx = np.zeros(len(hlives)+1)
    dydx[0] = -ln2/hlives[0]*y[0]
    for i in range(1, len(hlives)):
        dydx[i] = ln2/hlives[i-1]*y[i-1] - ln2/hlives[i]*y[i]
    dydx[-1] = ln2/hlives[-2]*y[-2]
    return dydx

#initial conditions and time interval
y0 = np.zeros(15)
y0[0] = 1 
x0 = 0
x1 = 4.46e9*365*24*60*60*10

#evaluate system of ODEs with an implicit method
ans = integrate.solve_ivp(decay_fun, (x0, x1), y0, method='Radau')

plt.figure()
plt.title('Uranium-238 decay chain vs. time')
for i in range(15):
    plt.plot(ans.t /(365*24*60*60), ans.y[i, :], label = labels[i])
plt.xlabel('t (years)')
plt.ylabel('Amount of isotope (arb. units)')
plt.legend()

plt.figure()
plt.title('Ratio of Pb206:U238 vs. time')
plt.plot(ans.t /(365*24*60*60), ans.y[-1, :]/ans.y[0,:] )
plt.xlabel('t (years)')
plt.ylabel('Pb206:U238')

#select a nicer time interval subset for the Th-230 : U-234 ratio
s = 10 #start index of answer for plotting
print('time (years)')
print(ans.t[s:] / (365*24*60*60))
print('ratio')
print(ans.y[4, s:]/ans.y[3, s:])

plt.figure()
plt.title('Ratio of Th230:U234 vs. time')
plt.plot(ans.t[s:] / (365*24*60*60), ans.y[4, s:]/ans.y[3, s:])
plt.xlabel('t (years)')
plt.ylabel('Pb206:U238')
plt.semilogx()
plt.show()




