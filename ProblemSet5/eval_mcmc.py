import numpy as np
import matplotlib.pyplot as plt
# from planck_likelihood import get_spectrum
# import camb
from scipy.ndimage import gaussian_filter

# def getchisq(data, errs, p):
#     pred = get_spectrum(p)
#     pred = pred[:len(data)]
#     residuals = (data - pred)
#     return np.sum((residuals/errs)**2)

#load data
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=2)
multipole = planck[:,0]
spectrum = planck[:,1]
specerrs = 0.5*(planck[:,2]+planck[:,3])

#load mcmc chain
chain = np.loadtxt('planck_chain.txt')

params, perrs = np.loadtxt('planck_fit_params.txt', usecols=(0,1), unpack=True) #use perrs to modulate step size.



# plt.figure()
# plt.plot(chain[:, 0])
# plt.title(r'MCMC $\chi^2$ over time')
# plt.xlabel('Iteration')
# plt.ylabel(r'$\chi^2$')

# #chains
titles = [r'$H_0$', r'$\Omega_b h^2$', r'$\Omega_c h^2$', r'$\tau$', r'$A_s$', r'$n_s$']
plt.subplots(2, 3)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(titles[i])
    plt.plot(chain[:, i+1])
    plt.xlabel('Iteration')
plt.subplot_tool()


s = 7000
plt.subplots(2, 3)
for i in range(6):
    plt.subplot(2, 3, i+1)
    ft = np.fft.rfft(chain[s:, i+1])
    power = np.abs(ft)**2
    freq = np.fft.rfftfreq(len(chain[s:, i+1]), )
    plt.plot(freq, power, '.')
    plt.semilogx()
    plt.semilogy()
    plt.title(titles[i])
plt.subplot_tool()
plt.show()


# print('parameters:')
# parameter_chain = chain[:, 1:]
# pfinal = np.mean(parameter_chain, axis=0)
# for val in pfinal:
#     print(val)
# print()
# print('errors')
# for er in np.std(parameter_chain, axis=0):
#     print(er)


# model = get_spectrum(pfinal)
# model = model[:len(spectrum)]
# print('chisq of mcmc', getchisq(spectrum, specerrs, pfinal))


# planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
# errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])

# plt.figure()
# plt.title("MCMC fit")
# plt.plot(multipole, model, label='Best fit model')
# plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.', label='Binned data')
# plt.xlabel('Multipole')
# plt.ylabel('Variance')
# plt.legend()
# plt.show()