import numpy as np
import matplotlib.pyplot as plt
import camb
from planck_likelihood import get_spectrum



def get_spec_var_choice(x, p, idx):
    p[idx] = x
    return(get_spectrum(p, lmax=3000))

#numerical derivative of function with respect to x 
def numpartial(fun, x, args, idx):
    dx = np.cbrt(1e-16)*x
    return ( fun( x+dx , args, idx) - fun( x-dx , args, idx) ) / (2*dx)

#numerical 
def newton_numerical(params):
    #fun is function being evaluated
    #t is independent variable
    #p is any other parameters we might need.

    #L is returned as the prediction given the parameters
    #grad is returned as the gradient.
    L = get_spectrum(params)
    grad = np.zeros((L.size, params.size))
    for i, p in enumerate(params):
        grad[:, i] = numpartial(get_spec_var_choice, p, params, i)
    return L, grad

#calculate chisq to see how it is changing
def getchisq(data, errs, pred):
    residuals = (data - pred)
    return np.sum((residuals/errs)**2)



planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=2)
multipole = planck[:,0]
spectrum = planck[:,1]
specerrs = 0.5*(planck[:,2]+planck[:,3])


#initial guess
p0 = np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
mag = 1
delta_chisq = 10. #set it to > 1 initially.
chisq = -10. #set to something that will definitely make first delta_chisq bigger than 1.
n_iter = 0
dp = np.zeros(p0.shape)

print("running newton's method")

#run newton's method:
while (delta_chisq > 1):
    p0 = p0 + dp
    n_iter += 1
    pred, grad = newton_numerical(p0)
    pred = pred[:len(spectrum)] #data is not same length as camb output.
    grad = grad[:len(spectrum)]

    #try with svd:
    u, s, v = np.linalg.svd(grad, 0)

    diff = (spectrum - pred).T

    dp = v.T @ np.diag(1/s) @ u.T @ diff

    newchi = getchisq(spectrum, specerrs, pred)
    delta_chisq = np.abs(chisq - newchi)
    chisq = newchi

    print()
    print(f'iteration {n_iter}')
    print('chisq', chisq)
    print('params')
    print(p0)

print('finished after ', n_iter, 'steps')


#error estimation:
rhs = grad.T @ np.diag(1/specerrs) @ grad
p_std = np.sqrt( np.diag( np.linalg.inv(rhs) ) )

output = np.zeros((p0.size, 2))
output[:, 0] = p0
output[:, 1] = p_std

np.savetxt('planck_fit_params.txt', output, )

#test parameter error 
ptest = p0 + p_std
model, grad = newton_numerical(ptest)
model = model[:len(spectrum)]
print('chisq after adding errors', getchisq(spectrum, specerrs, model))



planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])

plt.figure()
plt.title("Newton's method fit")
plt.plot(multipole, pred, label='Best fit model')
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.', label='Binned data')
plt.xlabel('Multipole')
plt.ylabel('Variance')
plt.legend()
plt.show()


