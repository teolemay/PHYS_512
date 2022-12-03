"""
PHY407: Lab 6, Question 2 (a)
Teophile Lemay

This script solves the motion of an vibrating N-storey building using a Verlet algorithm and a matrix representation for the system of ODEs.
"""

import numpy as np
import matplotlib.pyplot as plt


def make_matrix(N, km):
    """"
    this function makes an N-dimensional square A matrix as described in the lab 6 exercise sheet

    :param N: int, number of rows = number of cols 
    :param km: float, value of k/m

    :return: A = k/m * (array (-2 on diagonal, 1 beside the diagonal, 0 elsewhere))
    """
    array = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                array[i, j] = -2
            elif abs(i-j) == 1:
                array[i, j] = 1
    return km*array

def F(A, x):
    """
    this function returns calculates A*x (proper matrix-vector multiplication) for an NxN matrix A and vector x of length N. in this case, Equivalent to f(x) for 1 D case.
    """
    return np.dot(A, x)

#define parameters to use
km = 400 #rad/s
N = 10
dt = 0.001 #seconds
a = 0
b = 1

#create matrix
A_matrix = make_matrix(N, km)

#create time, x arrays
times = np.arange(a, b, dt)
x_vals = []

#initial positions (t=0) 
x_vec = np.zeros(N)
x_vec[0] = 0.1
#and initial velocity t=0+h/2
v_vec = 0.5*dt*F(A_matrix, x_vec)

for t in times:
    x_vals.append(x_vec)
    #update new x_vector using previous x position and v at next midpoint
    x_vec = x_vec + dt*v_vec #x(t+h) = x(t) + h*v(t + h/2)
    #update v to v(t + 3/2*h) to get next midpoint v ready.
    v_vec = v_vec + dt*F(A_matrix, x_vec) #here x_vec is already x(t + h)

x_vals = np.array(x_vals) #change to numpy array for consistency

#plot all motions on top of each other.
plt.figure()
for i in range(N):  
    ilabel = f'Floor {i}'
    #rgba colors: (red, green, blue, opacity). here, I iterate over different shades of blue.
    plt.plot(times, x_vals[:, i], color=(i/(N+0.1), i/(N+0.1), 1, 1), label=ilabel)
plt.xlabel('Time (s)')
plt.ylabel('x displacement (m)')
plt.legend()
plt.title('Simulated floor displacement over time for multiple storey building.')
plt.show()













