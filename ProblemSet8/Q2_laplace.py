import numpy as np
import matplotlib.pyplot as plt

def get_avg(A):
    V = A.copy()
    tot=np.roll(V,1,axis=0)+np.roll(V,-1,axis=0)+np.roll(V,1,axis=1)+np.roll(V,-1,axis=1)
    return tot/4

def make_rho(V, mask):
    V[mask] = 0
    ave = get_avg(V)
    ave[mask] = 0
    return V - ave

N = 201

m = N//2

dx = np.ones((N,N))*np.arange(0, N)
dx = np.abs(dx-m)**2
dy = dx.T

V_distance = np.sqrt(dx + dy)
V_distance[m,m] = 1 #placeholder to not get -inf for log(0)
V_pot = -(1/2/np.pi)*np.log(V_distance)
V_pot[m, m] = V_pot[m, m] = V_pot[m+1,m]*4 - V_pot[m+1,m-1] - V_pot[m+1,m+1] - V_pot[m+2,m]

for i in range(1000):
    V_pot = V_pot *1/V_pot[m,m]
    V_pot = get_avg(V_pot)
    V_pot[m, m] = V_pot[m, m] = V_pot[m+1,m]*4 - V_pot[m+1,m-1] - V_pot[m+1,m+1] - V_pot[m+2,m]

# rho = rho * 1/V_pot[m,m]
V_pot = V_pot *1/V_pot[m,m]

print()
print('V[1,0]:', V_pot[m+1, m])
print('V[2,0]:', V_pot[m+2, m])
print()
print('V[5,0]:', V_pot[m+5, m])
print()

#this is the Green's function!
Green = V_pot.copy()



#now I do the conjugate gradient part
#conjugate gradient algorithm:

def conj_grad_solver(mask, BC, eps=1e-16):
    #copying the algorithm from the notes:

    V_pot = np.zeros(mask.shape)

    RHS = get_avg(BC)
    RHS[mask] = 0

    r = RHS - make_rho(V_pot, mask)
    p = r.copy()
    rtr = np.sum(r**2)
    nsteps = 0
    while rtr > eps:
        nsteps += 1
        ap = make_rho(p, mask)
        pap=np.sum(p*ap)
        alpha = rtr/pap
        V_pot = V_pot + alpha*p
        r = r - alpha*ap
        rtr_new = np.sum(r**2)
        beta = rtr_new/rtr
        p = r + beta*p
        rtr = rtr_new

    print('convergence after ', nsteps, 'steps')
    V_pot[mask] = BC[mask]
    rho = V_pot - get_avg(V_pot)
    return V_pot, rho



mask = np.zeros((201, 201), dtype='bool')
mask[75:125, 75] = 1
mask[75:125, 125] = 1
mask[75, 75:125] = 1
mask[125, 75:125] = 1
mask[0, :] = 1
mask[-1, :] = 1
mask[:, 0] = 1
mask[:, -1] = 1

BC = mask.copy().astype(np.float64)
BC[0, :] = 0
BC[-1, :] = 0
BC[:, 0] = 0
BC[:, -1] = 0

V, rho = conj_grad_solver(mask, BC, eps=1e-20)

plt.figure()
plt.title('Charge density along side of box with constant $V$')
plt.plot(rho[70:130, 75])
plt.xlabel('x')
plt.ylabel('Charge density (arb. Units)')
plt.show()


#the potential field is the convolution of Green's function with the charge density.

Gfft = np.fft.rfft2(np.fft.fftshift(Green))
rhofft = np.fft.rfft2(rho)
V_field = np.fft.irfft2(Gfft * rhofft)

plt.figure()
plt.title('Potential field over all space')
plt.imshow(V_field)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

plt.figure()
plt.title('Potential field inside box')
plt.imshow(V_field[76:124, 76:124])
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()


xvec = V_field - np.roll(V_field, 1, axis=1)
yvec = V_field - np.roll(V_field, 1, axis=0)

plt.figure()
plt.title('Potential field gradient')
plt.quiver(xvec[::2, ::2], yvec[::2, ::2])
plt.xlabel('x')
plt.ylabel('y')
plt.xticks([])
plt.yticks([])
plt.show()



