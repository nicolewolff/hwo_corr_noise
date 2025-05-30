import emcee
# import corner
import h5py
import sys
from prospect.plotting import corner
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.lines  as mlines
from astropy.table       import Table, Column, MaskedColumn
from astropy.io          import ascii
from rfast_routines      import spectral_grid
from rfast_routines      import gen_spec
from rfast_routines      import kernel_convol
from rfast_routines      import gen_spec_grid
from rfast_routines      import inputs
from rfast_routines      import init
from rfast_routines      import init_3d
from rfast_atm_routines  import set_gas_info
from rfast_atm_routines  import setup_atm
from rfast_atm_routines  import mmr2vmr
from rfast_atm_routines  import vmr2mmr
from rfast_opac_routines import opacities_info
from rfast_user_models   import surfalb
from rfast_user_models   import cloud_optprops
from rfast_user_models   import cloud_struct
from helper_scripts      import return_names_truths_etc, reademceeh5, plot_covariance_matrix, plot_true_vs_inferred

nburn, thin, names, ndim, truths = return_names_truths_etc()

def overlay_corner_plots(num_iterations, snr_uncorr, snr_tot, length):
  snr_uncorr_str = "{:.2f}".format(snr_uncorr)
  snr_tot_str = "{:.2f}".format(snr_tot)

  all_samples_uncorr = []
  all_samples_corr = []

  for i in range(num_iterations):
    samples_corr, _ = reademceeh5('outputs_correlated_052325/snr_uncorr_'+snr_uncorr_str+'_snr_tot_'+snr_tot_str+'_length_'+length+'/'+str(i+1)+'/earth_refl_demo_snr20_14dim.h5',nburn,thin)
    all_samples_corr.append(samples_corr.reshape((-1, ndim)))
    samples_uncorr, _ = reademceeh5('outputs_uncorrelated/snr_'+str(snr_tot)+'_'+str(i+1)+'/earth_refl_demo_snr20_14dim.h5',nburn,thin)
    all_samples_uncorr.append(samples_uncorr.reshape((-1,ndim)))

  all_samples_corr = np.vstack(all_samples_corr) 
  all_samples_uncorr = np.vstack(all_samples_uncorr)

  # all_samples_uncorr = np.array(all_samples_uncorr)
  # all_samples_corr = np.array(all_samples_corr)
  avg_uncorr = all_samples_uncorr
  avg_corr = all_samples_corr

  ## Prospect corner plot ##
  cfig, axes = plt.subplots(ndim, ndim, figsize=(10,9))
  axes = corner.allcorner(all_samples_uncorr.T, names, axes, color="black")
  corner.allcorner(all_samples_corr.T, names, axes, color="royalblue")
  for j in range(len(avg_uncorr.T)):
    axes[j,j].axvline(truths[j], lw=2, color='black')
    median_uncorr = np.percentile(avg_uncorr[:, j], 50)
    median_corr = np.percentile(avg_corr[:, j], 50)

    lo_uncorr, hi_uncorr = np.percentile(avg_uncorr[:, j], [16, 84])
    lo_corr, hi_corr = np.percentile(avg_corr[:, j], [16, 84])

    param_name = names[j]
    title_str = r"{}: {:.2f} $\pm^{{+{:.2f}}}_{{-{:.2f}}}$".format(
          param_name, median_uncorr, hi_uncorr - median_uncorr, median_uncorr - lo_uncorr
          )+"\n"+r"{}: {:.2f} $\pm^{{+{:.2f}}}_{{-{:.2f}}}$".format(
          param_name, median_corr, hi_corr - median_corr, median_corr - lo_corr)   
    axes[j,j].set_title(title_str, fontsize=8)

  cfig.suptitle('SNR '+snr_tot_str+', Uncorrelated (black) vs Correlated (blue), '+str(num_iterations)+' noise instances', fontsize=18)
  cfig.savefig('plots/high_res_averaging_snrtot_'+snr_tot_str+'_corr_vs_uncorr_'+length+'_corner.png',format='png',bbox_inches='tight')
  plt.close(cfig)
  return 

overlay_corner_plots(num_iterations=4, snr_uncorr=28.28, snr_tot=14.14, length='4e-3')
# plot_true_vs_inferred(num_random=4, snr_uncorr=28.28, snr_tot=14.14, length='4e-3')
# plot_covariance_matrix('7_quick_check.png')