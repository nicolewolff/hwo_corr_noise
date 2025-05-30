from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

mpl.rc('xtick.major', size=5, pad=3, width=2)
mpl.rc('ytick.major', size=5, pad=3, width=2)
mpl.rc('xtick.minor', size=2, pad=3, width=2)
mpl.rc('ytick.minor', size=2, pad=3, width=2)
mpl.rc('axes', linewidth=2, labelsize=14, titlesize=18)
mpl.rc('legend', fontsize=14)
mpl.rc('lines', markersize=5)
mpl.rc('font', size=12)
cmap = 'magma'

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
num_iterations = 4
fig, ax = plt.subplots(2, 1)

### Correlated ###
for i in range(num_iterations):

    data        = ascii.read('outputs_correlated_052325/snr_uncorr_28.28_snr_tot_7.07_length_4e-3/'+str(i+1)+'/earth_refl_demo_snr20_cnse.dat',data_start=1,delimiter='|')
    lam         = data['col2'][:]
    dlam        = data['col3'][:]
    dat         = data['col6'][:]  # want to plot dat
    err         = data['col7'][:]

    # plt.errorbar(lam, data, yerr=err, fmt=".k")
    ax[1].plot(lam, dat, drawstyle='steps-mid', color=colors[i])

ax[1].set_xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
# ax[1].set_xlim(0.4,1.0)

### Uncorrelated ###
for i in range(num_iterations):

    data        = ascii.read('outputs_uncorrelated/snr_7.07_'+str(i+1)+'/earth_refl_demo_snr20_cnse.dat',data_start=1,delimiter='|')
    lam         = data['col2'][:]
    dlam        = data['col3'][:]
    dat         = data['col6'][:]  # want to plot dat
    err         = data['col7'][:]

    # plt.errorbar(lam, data, yerr=err, fmt=".k")
    ax[0].plot(lam, dat, drawstyle='steps-mid', color=colors[i])

ax[0].set_xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
# ax[0].set_xlim(0.4,1.0)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.ylabel('Planet-to-Star Flux Ratio')

fig.suptitle('SNR 7, Uncorrelated vs Correlated')
plt.savefig('plots/spectra_snr_7_update.png',format='png',bbox_inches='tight')
plt.close()