import numpy as np
import matplotlib.pyplot as plt
import camb
from planck_likelihood import get_spectrum


#calculate chisq to see how it is changing
def getchisq(data, errs, p):
    pred = get_spectrum(p)
    pred = pred[:len(data)]
    residuals = (data - pred)
    return np.sum((residuals/errs)**2)


planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=2)
multipole = planck[:,0]
spectrum = planck[:,1]
specerrs = 0.5*(planck[:,2]+planck[:,3])

params, perrs = np.loadtxt('planck_fit_params.txt', usecols=(0,1), unpack=True) #use perrs to modulate step size.

nsteps = 20000
nchains = 1 #just doing 1 long chain.

print('running mcmc ...')
chain = np.zeros((nsteps, len(params)+1)) #first column for chi values.
chain[0, 1:] = np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0]) #old parameters from 
chi = getchisq(spectrum, specerrs, chain[0, 1:]) #set initial chisq to be kinda bad.
chain[0, 0] = chi
for j in range(1, nsteps):
    newparams = chain[j-1, 1:] + np.random.randn(len(perrs))*perrs
    chi_new = getchisq(spectrum, specerrs, newparams)

    if chi_new < chi:
        chi = chi_new
        chain[j, 1:] = newparams
    else:
        accept = np.exp(-0.5*(chi_new - chi))
        if accept > np.random.rand():
            chain[j, 1:] = newparams
            chi = chi_new
        else:
            chain[j, 1:] = chain[j-1, 1:]
    chain[j, 0] = chi
    print(f'step {j}, chisq = {chi}', end='\r')

np.savetxt('planck_chain.txt', chain)

print(" "*20, end='\r')

print('parameters:')
parameter_chain = chain[:, 1:]
pfinal = np.mean(parameter_chain, axis=1)
for val in pfinal:
    print(val)
print()
print('errors')
for er in np.std(parameter_chain, axis=1):
    print(er)




model = get_spectrum(pfinal)
model = model[:len(spectrum)]
print('chisq after adding errors', getchisq(spectrum, specerrs, pfinal))


planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])

plt.figure()
plt.title("MCMC fit")
plt.plot(multipole, model, label='Best fit model')
plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.', label='Binned data')
plt.xlabel('Multipole')
plt.ylabel('Variance')
plt.legend()
plt.show()