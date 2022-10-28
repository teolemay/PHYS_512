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

params, perrs = np.loadtxt('planck_fit_params.txt', usecols=(0,1), unpack=True) #use perrs from newton fit to modulate step size for mcmc

nsteps = 40000

print('running mcmc ...')
chain = np.zeros((nsteps, len(params)+1)) #first column for chi values.
chain[0, 1:] = np.asarray([60,0.02,0.1,0.054,2.00e-9,1.0]) #start with some bad parameters
chi = getchisq(spectrum, specerrs, chain[0, 1:]) #set initial chisq to be kinda bad.
chain[0, 0] = chi
for j in range(1, nsteps):
    good_tau=False
    count = 0
    while not good_tau:
        count += 1
        mod = np.random.randn(len(perrs))
        newparams = chain[j-1, 1:] + mod*perrs
        if (newparams[3] > (0.054 - 0.0074)) and ( newparams[3] < (0.054 + 0.0074)):
            good_tau=True

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
    print(f'step {j}, took {count} tries to get params, chisq = {chi}', end='\r')

print(' '*40)
print('all done!')
np.savetxt('planck_chain_tauprior.txt', chain)



