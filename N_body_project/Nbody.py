import numpy as np
import matplotlib.pyplot as plt
import numba as nb

#This function is directly copied (with some notation changes) from "nbody_nb_fft.py".
@nb.njit(parallel=True)
def assign_grad(x, y, potential, gradx, grady, n):
    for i in nb.prange(x.shape[0]):
        if x[i] < 0:
            xidx=n-1
            xidxa=0
            px = x[i] + 1
        else:
            xidx = int(x[i])
            xidxa = xidx+1 #idx (A)bove since int() always rounds down.
            if xidxa==n:
                xidxa=0
            px = x[i] - xidx
        if y[i] < 0:
            yidx=n-1
            yidxa = 0
            py = y[i] + 1
        else:
            yidx = int(y[i])
            yidxa = yidx+1
            if yidxa==n:
                yidxa=0
            py = y[i] - yidx
        gradx[i] = (potential[xidxa, yidx] - potential[xidx, yidx])*(1-py) + (potential[xidxa, yidxa] - potential[xidx, yidxa])*(py)
        grady[i] = (potential[xidx, yidxa] - potential[xidx, yidx])*(1-px) + (potential[xidxa, yidxa] - potential[xidxa, yidx])*(px)  



class particles():
    def __init__(self, N=2, xy=None, vxy=None, Ngrid=1000, soft=1, dt=0.001, M=None, periodic=True):
        self.periodic=periodic
        self.soft=soft
        self.N = N
        self.Ngrid = Ngrid
        self.dt = dt
        if hasattr(M, '__len__'):
            self.mass = M
        else:
            self.mass = np.ones(N)
        self.x = np.array(xy[:, 0])
        self.y = np.array(xy[:, 1])
        self.vx = vxy[:, 0]
        self.vy = vxy[:, 1]
        self.oldx = self.x.copy()
        self.oldy = self.y.copy()
        self.oldvx = self.vx.copy()
        self.oldvy = self.vy.copy()
        
        self.gradx = np.zeros(len(self.x))
        self.grady = np.zeros(len(self.x))

        self.density = np.zeros((2*self.Ngrid, 2*self.Ngrid))
        self.U = None
        self.kernelft = None  #potential for a single particle
        self.p = 0
        self.k = 0
        self.t = 0
    #this function is also mostly identical to one in "nbody_nb_fft.py"
    def get_kernelft(self):
        if self.periodic:
            n = self.Ngrid
        else:
            n = 2*self.Ngrid
        x=np.fft.fftfreq(n)*n
        rsqr=np.outer(np.ones(n),x**2)
        rsqr=rsqr+rsqr.T
        rsqr[rsqr<self.soft**2]=self.soft**2
        kernel=rsqr**-0.5
        self.kernelft = np.fft.rfft2(kernel)

    def get_density(self):
        if self.periodic:
            #int(-(1,0]) (not inclusive for 1) will be 0. therefore, only wrap things that should actually round to the other side.
            #for consistency, do same 0.5 past max value on the other end.
            #use modulo operators for possible long range BC -> if you are multiple Ngrid awway, still included by remainder of removing all the multiples of Ngrid
            self.x[self.x > self.Ngrid-0.5] = self.x[self.x > self.Ngrid-0.5] % (self.Ngrid-1)
            self.x[self.x < -0.5] = self.Ngrid - (-self.x[self.x < -0.5] % (self.Ngrid-1))
            self.y[self.y > self.Ngrid-0.5] = self.y[self.y > self.Ngrid-0.5] % (self.Ngrid-1)
            self.y[self.y < -0.5] = self.Ngrid - (-self.y[self.y < -0.5] % (self.Ngrid-1))
        else:
            x_right = self.x > self.Ngrid-0.5
            x_left = self.x < -0.5
            y_top = self.y > self.Ngrid-0.5
            y_bot = self.y < -0.5
            x_violation = np.logical_or(x_right, x_left)
            y_violation = np.logical_or(y_top, y_bot)
            remove = np.logical_or(x_violation, y_violation)
            self.x = np.delete(self.x, remove) 
            self.y = np.delete(self.y, remove) 
            self.mass = np.delete(self.mass, remove) 
            self.vx = np.delete(self.vx, remove)
            self.vy = np.delete(self.vy, remove)
            self.gradx = np.delete(self.gradx, remove)
            self.grady = np.delete(self.grady, remove)
            #remove particles outside boundary.
        self.density, nope, nothanks = np.histogram2d(self.x, self.y, self.Ngrid, weights=self.mass, range=[[-0.5, self.Ngrid-0.5], [-0.5, self.Ngrid-0.5]])

    def get_potential(self):
        self.U = np.fft.irfft2(np.fft.rfft2(self.density) * self.kernelft, (self.Ngrid, self.Ngrid))

    def get_grad(self):
        assign_grad(self.x, self.y, self.U, self.gradx, self.grady, self.Ngrid)

    def do_step(self):
        # copied step taking function form from the "nbody_nb_fft.py" this is clearly a euler integrator step 
        # and I keep it here just for testing.
        self.get_density()
        self.get_potential()
        self.get_grad()
        #update position
        self.x = self.x + self.vx*self.dt 
        self.y = self.y + self.vy*self.dt 
        #update velocities
        self.vx = self.vx + self.gradx*self.dt
        self.vy = self.vy + self.grady*self.dt
        #need something to check the energies
        self.p = np.sum(self.U*self.density)
        self.k = np.sum(self.vx**2 + self.vy**2)
    def leapfrog_step(self):
        #re-writing this because I think I had something wrong before.
        self.get_density()
        self.get_potential()
        self.get_grad()
        if self.t == 0: #if this is the first
            #initial half step
            self.get_density()
            self.get_potential()
            self.get_grad()
            self.vx = self.vx + self.gradx*self.dt*0.5
            self.vy = self.vy + self.grady*self.dt*0.5
            self.x = self.x + self.vx*self.dt*0.5
            self.y = self.y + self.vy*self.dt*0.5
        else:
            #take a full step, using gradient from the half step.
            self.get_density()
            self.get_potential()
            self.get_grad()
            newvx = self.oldvx + self.gradx*self.dt
            newvy = self.oldvy + self.grady*self.dt
            self.oldvx = self.vx.copy()
            self.oldvy = self.vy.copy()
            self.vx = newvx
            self.vy = newvy
            newx = self.oldx + self.vx*self.dt
            newy = self.oldy + self.vy*self.dt
            self.oldx = self.x.copy()
            self.oldy = self.y.copy()
            self.x = newx
            self.y = newy
        self.t = 1 # just shut off the initial step thing.
        self.p = np.sum(self.U*self.density)
        self.k = np.sum((self.vx**2 + self.vy**2)*self.mass)

    def rk4_step(self):
        def get_derivs(xx):
        nn=xx.shape[0]//2
        x=xx[:nn,:]
        v=xx[nn:,:]
        density = self.get_density(x, y)
        U = self.get_potential(density)
        gx, gy = self.get_grad(U)
        return np.vstack([v, gx, gy])
    def take_step_rk4(x,v,dt):
        xx=np.vstack([x,v])
        k1=get_derivs(xx)
        k2=get_derivs(xx+k1*dt/2)
        k3=get_derivs(xx+k2*dt/2)
        k4=get_derivs(xx+k3*dt)
        
        tot=(k1+2*k2+2*k3+k4)/6
        

        nn=x.shape[0]
        x=x+tot[:nn,:]*dt
        v=v+tot[nn:,:]*dt
        return x,v




    
N = 1000000
n = 1000
xy = np.zeros((N, 2))
vxy = np.zeros((N, 2))

# xy[:, :] = np.random.randn(N, 2)*n/12+n/2
# xy[N//2:, 0] = xy[N//2:, 0] - n/5
# xy[:N//2, 0] = xy[:N//2, 0] + n/5
# vxy[:n, 1] = 25
# vxy[n:] = -25
xy[:, :] = np.random.rand(N, 2)*n 
vxy[:N, 1] = 25
vxy[N:, 1] = -25



p = particles(N=N, xy=xy, vxy=vxy, Ngrid=n, dt=0.02, soft=2, periodic=True)
p.get_kernelft()
p.get_density()
p.get_potential()

###################################### simulation below


plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
im=ax.imshow(p.density[:, :]**0.5)
# im = ax.imshow(p.U)

# ax.plot(p.x[0], p.y[0], '.')
# ax.plot(p.x[1], p.y[1], '.')
# ax.set_xlim(460, 540)
# ax.set_ylim(460, 540)


T = 10000
pE = np.zeros(T)
kE = np.zeros(T)
for i in range(T):
    # p.leapfrog_step()
    p.do_step()
    pE[i] = p.p
    kE[i] = p.k
    if i%20 == 0:
        print(i, ':', p.p, p.k, p.p+p.k)
        # print('y', p.grady, 'x', p.gradx)
        # print(p.vy)
    # xpos.append(p.x[0])
    # ypos.append(p.y[0])
        im.set_data(p.density[:, :]**0.5)
        # im.set_data(p.U)
        # print(p.p, p.k, p.p+p.k)
        # ax.plot(p.x[0], p.y[0], '.')
        # ax.plot(p.x[1], p.y[1], '.')
        # ax.set_xlim(460, 540)
        # ax.set_ylim(460, 540)
        plt.pause(0.001)

np.save('periodic_potential.npy', pE)
np.save('periodic_kinetic.npy', kE)
        