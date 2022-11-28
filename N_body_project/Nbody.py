import numpy as np
import matplotlib.pyplot as plt
import numba as nb


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
        # if xy == None:
        #     self.x = np.arange(1, N+1)
        #     self.y = np.arange(1, N+1)
        self.x = np.array(xy[:, 0])
        self.y = np.array(xy[:, 1])
        self.vx = vxy[:, 0]
        self.vy = vxy[:, 1]
        
        self.gradx = np.zeros(len(self.x))
        self.grady = np.zeros(len(self.x))

        self.density = np.zeros((self.Ngrid, self.Ngrid))
        self.U = None
        self.kernelft = None  #potential for a single particle
        self.p = 0
        self.k = 0

    def get_kernelft(self):
        # # hardcoded soft=1 implementation.
        if self.periodic:
            n = self.Ngrid
        else:
            n = 2*self.Ngrid
        # a = np.ones((n,n))
        # for i in range(n):
        #     for j in range(n):
        #         if not (i == n//2 and j == n//2):
        #             a[i, j] =  1/np.sqrt((i-n//2)**2 + (j-n//2)**2)
        # self.kernelft = np.fft.rfft2(a)
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
            x_violation = np.logical_or(self.x > self.Ngrid-0.5, self.x < -0.5)
            y_violation = np.logical_or(self.y > self.Ngrid-0.5, self.y < -0.5)
            remove = np.logical_or(x_violation, y_violation)
            self.x = np.delete(self.x, remove)
            self.y = np.delete(self.y, remove)
            self.m = np.delete(self.m, remove) 
            self.vx = np.delete(self.vx, remove)
            self.vy = np.delete(self.vy, remove)
            self.gradx = np.delete(self.gradx, remove)
            self.grady = np.delete(self.grady, remove)
            #remove particles outside boundary
        H, nope, nothanks = np.histogram2d(self.x, self.y, self.Ngrid, weights=self.mass, range=[[-0.5, self.Ngrid-0.5], [-0.5, self.Ngrid-0.5]])
        self.density = H

    def get_potential(self):
        self.U = np.fft.irfft2(np.fft.rfft2(self.density) * self.kernelft, (self.Ngrid, self.Ngrid))

    def get_grad(self):
        # grad = np.gradient(self.U, 1)
        assign_grad(self.x, self.y, self.U, self.gradx, self.grady, self.Ngrid)
        # assign_grad(self.x, self.y, grad, self.gradx, self.grady, self.Ngrid)

    def do_step(self):
        # print('step')
        #this is definitely a euler integrator step.
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
        #half position step
        self.x = self.x + self.vx*self.dt*0.5
        self.y = self.y + self.vy*self.dt*0.5
        #calculate change in velocity at half step position
        self.get_density()
        self.get_potential()
        self.get_grad()
        self.vx = self.vx + self.gradx*self.dt
        self.vy = self.vy + self.grady*self.dt
        #finish position step with new velocities
        self.x = self.x + self.vx*self.dt*0.5
        self.y = self.y + self.vy*self.dt*0.5
        self.p = np.sum(self.U*self.density)
        self.k = np.sum(self.vx**2 + self.vy**2)
        # print('step')

    
N = 2
n = 1000
xy = np.zeros((N, 2))
vxy = np.zeros((N, 2))

xy[0, :] = 499.6, 475
xy[1, :] = 500.4, 525
vxy[0, :] = 2, 0
vxy[1, :] = -2, 0

M = np.array([400,400])

p = particles(N=N, xy=xy, vxy=vxy, Ngrid=n, M=M, dt=0.01, soft=2)
print(len(p.mass))
p.get_kernelft()
p.get_density()
p.get_potential()
# # ##########################  testing some things
# p.get_potential()
# p.leapfrog_step()

# plt.figure()
# plt.imshow(p.U[490:510, 480:520])
# # plt.plot(p.U[498, :])
# # plt.xlim(450, 560)
# # plt.figure()
# # plt.plot(p.U[499, :])
# # plt.xlim(450, 560)
# plt.figure()
# plt.plot(p.U[500, :])
# plt.xlim(450, 560)
# # plt.figure()
# # plt.plot(p.U[501, :])
# # plt.xlim(450, 560)
# # plt.figure()
# # plt.plot(p.U[502, :])
# # plt.xlim(450, 560)
# plt.show()


###################################### simulation below


# plt.ion()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # im=ax.imshow(p.density[:, :]**0.5)

# ax.plot(p.x[0], p.y[0], '.')
# ax.plot(p.x[1], p.y[1], '.')
# ax.set_xlim(460, 540)
# ax.set_ylim(460, 540)


T = 50000
x = np.zeros((T, 2))
y = np.zeros((T, 2))
for i in range(T):
    # p.do_step()
    # print(p.vy)
    p.leapfrog_step()
    x[i, :] = p.x
    y[i, :] = p.y
    if i%100 == 0:
        print(i)
        # print('y', p.grady, 'x', p.gradx)
        # print(p.vy)
    # xpos.append(p.x[0])
    # ypos.append(p.y[0])
        # im.set_data(p.density[:, :]**0.5)
        # print(p.p, p.k, p.p+p.k)
        # ax.plot(p.x[0], p.y[0], '.')
        # ax.plot(p.x[1], p.y[1], '.')
        # ax.set_xlim(460, 540)
        # ax.set_ylim(460, 540)
        # plt.pause(0.001)
        
# plt.figure()
# plt.title('1 particle x-position over time')
# plt.plot(range(len(xpos)), xpos)
# plt.ylabel('x')
# plt.xlabel('Steps')
# plt.figure()
# plt.title('1 particle y-position over time')
# plt.plot(range(len(ypos)), ypos)
# plt.ylabel('y')
# plt.xlabel('Steps')
# plt.show()
# print(xpos)
# print(ypos)

np.save(r'C:\Users\teole\anaconda3\envs\coursework\phys512\PHYS_512\N_body_project\xpos.npy', x)
np.save(r'C:\Users\teole\anaconda3\envs\coursework\phys512\PHYS_512\N_body_project\ypos.npy', y)
        