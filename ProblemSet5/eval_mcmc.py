import numpy as np
import matplotlib.pyplot as plt
from planck_likelihood import get_spectrum
import camb

def getchisq(data, errs, p):
    pred = get_spectrum(p)
    pred = pred[:len(data)]
    residuals = (data - pred)
    return np.sum((residuals/errs)**2)

#load data
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=2)
multipole = planck[:,0]
spectrum = planck[:,1]
specerrs = 0.5*(planck[:,2]+planck[:,3])
#for plotting
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])

#load mcmc chain
chain = np.loadtxt('planck_chain.txt')

#show chain evolution
start = 20000 #burn in removal index.
#chains
titles = [r'$H_0$', r'$\Omega_b h^2$', r'$\Omega_c h^2$', r'$\tau$', r'$A_s$', r'$n_s$']
plt.subplots(2, 3)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(titles[i])
    plt.plot(chain[:, i+1])
    plt.xlabel('Iteration')
plt.subplot_tool()

# #show chain power spectrums
plt.subplots(2, 3)
for i in range(6):
    plt.subplot(2, 3, i+1)
    ft = np.fft.rfft(chain[start:, i+1])
    power = np.abs(ft)**2
    freq = np.fft.rfftfreq(len(chain[start:, i+1]), )
    plt.plot(freq, power, linewidth=1)
    plt.semilogx()
    plt.semilogy()
    plt.title(titles[i])
plt.subplot_tool()

#show chi squared over chain
plt.figure()
plt.plot(chain[start:, 0])
plt.title(r'MCMC $\chi^2$')
plt.xlabel('Steps')
plt.ylabel(r'$\chi^2$')


# show corner plots
plt.subplots(6,6)
for i in range(6): #which row
    for j in range(6): #which collumn
        plt.subplot(6,6, (i*6)+j+1 )
        if i == j:
            plt.hist(chain[start:, i+1], bins=20)
        else:
            plt.scatter(chain[start:, j+1], chain[start:, i+1], s=1, c='k', marker='.')
        if j != 0:
            plt.yticks([])
        if i != 5:
            plt.xticks([])
        if j == 0:
            plt.ylabel(titles[i])
        if i == 5:
            plt.xlabel(titles[j])
            plt.xticks(rotation=45, ha='right')
plt.show()


#show raw chain parameters (burn in removed)
print('parameters:')
parameter_chain = chain[start:, 1:]
pfinal = np.mean(parameter_chain, axis=0)
errs = np.std(parameter_chain, axis=0)
for i, val in enumerate(pfinal):
    print(titles[i], val, '(+/-)', errs[i])
print()


#show plot of model with raw chain parameters 
model = get_spectrum(pfinal)
model = model[:len(spectrum)]
print('chisq of mcmc: ', getchisq(spectrum, specerrs, pfinal))


#importance sampling
p_tau= pfinal.copy()
p_tau[3] = 0.054 #adjust tau to the constrained parameter value.
tau_chi = getchisq(spectrum, specerrs, p_tau)
print('chisq for mcmc with swapped tau: ', tau_chi)

################################

# Re-calculate parameters using a weighted average.
# For each step in the chain, weight = exp(-0.5 (chisq - tau_chi))
# weighted average = sum( weight * value)/sum(weights)

# code below is my attempt at importance sampling. Unfortunately, chi squared is much worse if tau is swapped to the constrained value, so 
# defining weights using exp(-0.5 * delta chi squared) produces an overflow error. 

###############################

# weights = np.exp(-0.5*(chain[start:, 0] - tau_chi))
# p_weighted = np.average(chain[start:, 1:], axis=0, weights=weights)

# weighted_std_dev = np.sqrt( np.sum(weights*(chain[start:, 1:] - p_weighted)**2, axis=0) / (np.sum(weights, axis=0)) )

# print('importance sampling parameters:')
# for i, val in enumerate(p_weighted):
#     print(titles[i], val, '(+/-)', weighted_std_dev[i])
# print()

# #show plot of model with importance sampling parameters 
# model = get_spectrum(p_weighted)
# model = model[:len(spectrum)]
# print('chisq of importance sampling: ', getchisq(spectrum, specerrs, p_weighted))

