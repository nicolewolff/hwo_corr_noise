from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np


colors = [
    "xkcd:blush pink",
    "xkcd:lipstick red",
    "xkcd:reddy brown",
    "xkcd:shamrock",
    "lightseagreen",
    "xkcd:peacock blue",
    "xkcd:denim blue",
    "xkcd:light indigo",
    "xkcd:rose"
]

### Correlated ###
num_iterations = 9
for i in range(num_iterations):

    data        = ascii.read('outputs_correlated_new/snr_uncorr_28.28_snr_tot_7.07_length_4e-3/'+str(i+1)+'/earth_refl_demo_snr20_cnse.dat',data_start=1,delimiter='|')
    lam         = data['col2'][:]
    dlam        = data['col3'][:]
    dat         = data['col6'][:]  # want to plot dat
    err         = data['col7'][:]

    # plt.errorbar(lam, data, yerr=err, fmt=".k")
    plt.plot(lam, dat, drawstyle='steps-mid', color=colors[i])

data        = ascii.read('outputs_correlated_new/snr_uncorr_28.28_snr_tot_7.07_length_4e-2/1/earth_refl_demo_snr20_cnse.dat',data_start=1,delimiter='|')
lam         = data['col2'][:]
dlam        = data['col3'][:]
dat         = data['col6'][:]  # want to plot dat
err         = data['col7'][:]
plt.plot(lam, dat, drawstyle='steps-mid', color='black')

plt.ylabel('Planet-to-Star Flux Ratio')
plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
plt.savefig('plots/correlated_spectra.png',format='png',bbox_inches='tight')
plt.close()

### Uncorrelated ###
num_iterations = 9
for i in range(num_iterations):

    data        = ascii.read('outputs_uncorrelated/snr_7.07_'+str(i+1)+'/earth_refl_demo_snr20_cnse.dat',data_start=1,delimiter='|')
    lam         = data['col2'][:]
    dlam        = data['col3'][:]
    dat         = data['col6'][:]  # want to plot dat
    err         = data['col7'][:]

    # plt.errorbar(lam, data, yerr=err, fmt=".k")
    plt.plot(lam, dat, drawstyle='steps-mid', color=colors[i])

plt.ylabel('Planet-to-Star Flux Ratio')
plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
plt.savefig('plots/uncorrelated_spectra.png',format='png',bbox_inches='tight')
plt.close()