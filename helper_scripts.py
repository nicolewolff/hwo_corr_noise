import emcee
# import corner
import h5py
import sys
import os
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

def reademceeh5(fn,nburn,thin,flatten=False):
  # open file, important data
  hf       = h5py.File(fn,'r')
  grps     = [item for item in hf['mcmc'].values()]
  # extract samples chain and log-likelihood, remove burn-in
  if (nburn >= 0):
    samples  = grps[1][nburn:,:,:]
    lnprob   = grps[2][nburn:,:]
  else:
    samples  = grps[1][nburn:,:,:]
    lnprob   = grps[2][nburn:,:]
  # thin
  samples  = samples[0::thin,:,:]
  lnprob   = lnprob[0::thin,:]
  # flatten
  if flatten:
    samples  = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
    lnprob   = lnprob.reshape(lnprob.shape[0]*lnprob.shape[1])
  # close h5 file
  hf.close()
  return samples,lnprob
  
def return_names_truths_etc():
    # filename_scr = input("rfast inputs script filename: ") # otherwise ask for filename
    filename_scr = 'outputs_correlated_debugging/snr_14.14_0.5correlated_length_4e-3/1/rfast_inputs.scr'
    sys.argv.append(filename_scr) # poor practice, but prevents bug when importing Fx

    # obtain input parameters from script
    fnr,fnn,fns,dirout,Nlev,pmin,pmax,bg,\
    species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,mmri,\
    tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,\
    species_l,species_c,\
    lams,laml,res,regrid,smpl,opdir,\
    Rp,Mp,gp,a,Apars,em,\
    cld,phfc,opars,cpars,lamc0,fc,\
    ray,ref,sct,fixp,pf,fixt,tf,p10,fp10,\
    src,\
    alpha,ntg,\
    Ts,Rs,\
    ntype,snr0,lam0,rnd,correlated,inclcovar,sigmarho,arho,\
    clr,fmin,mmrr,nwalkers,nstep,nburn,thin,restart,progress = inputs(filename_scr)

    # input data filename
    fn_dat = fnn + '.dat'

    # set info for all radiatively active gases
    Ngas,gasid,mmw0,ray0,nu0,mb,rayb = set_gas_info(bg)

    # get initial gas mixing ratios
    p,t,t0,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
                                        tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                        species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                        mmri,mb,Mp,Rp,p10,fp10,src,ref,nu0)

    # convert between mmr and vmr, if needed
    if (mmri != mmrr):
        if mmri: # convert input mmr to vmr
            f,fb,f0 = mmr2vmr(mmw0,gasid,species_r,m,mb,f0,fb,f)
        else: # otherwise convert input vmr to mmr
            f,fb,f0 = vmr2mmr(mmw0,gasid,species_r,m,mb,f0,fb,f)

    # read input data
    data        = ascii.read(dirout+fn_dat,data_start=1,delimiter='|')
    lam         = data['col2'][:]
    dlam        = data['col3'][:]
    dat         = data['col6'][:]
    err         = data['col7'][:]

    # save input radius for thermal emission case
    Rpi = Rp

    # generate wavelength grids
    Nres             = 3 # no. of res elements to extend beyond grid edges to avoid edge sensitivity (2--4)
    if regrid:
        lam_hr,dlam_hr = gen_spec_grid(lams,laml,np.float_(res)*smpl,Nres=np.rint(Nres*smpl))
    else:
        x_low = min(0.01,min(lam)-dlam[0]*Nres) # note: prevent min wavelength of 0 um
        x_hgh = max(lam)+dlam[-1]*Nres
        lam_hr,dlam_hr = spectral_grid(x_low,x_hgh,res=lam/dlam*smpl,lamr=lam)

    # assign photometric vs. spectroscopic points
    mode           = np.copy(lam_hr)
    mode[:]        = 1

    # inform user of key opacities information
    opacities_info(opdir)

    # initialize opacities and convolution kernels
    sigma_interp,cia_interp,ncia,ciaid,kern = init(lam,dlam,lam_hr,species_l,species_c,opdir,pf,tf,mode=mode)

    # surface albedo model
    As = surfalb(Apars,lam_hr)

    # cloud optical properties: asymmetry parameter, single scattering albedo, extinction efficiency
    gc,wc,Qc = cloud_optprops(opars,cld,opdir,lam_hr)

    # cloud vertical structure model
    dtauc0,atm = cloud_struct(cpars,cld,p,t,z,grav,f,fb,m)
    p,t,z,grav,f,fb,m = atm

    # initialize disk integration quantities
    threeD   = init_3d(src,ntg)

    # package parameters for user-defined routines
    tiso = tpars[0]
    A0 = Apars[0]
    # no parameters for cloud optical properties model
    pt,dpc,tauc0 = cpars

    # parameter names
    lfO2,lfH2O,lfCO2 = np.log10(f0[species_r=='o2'])[0],np.log10(f0[species_r=='h2o'])[0],np.log10(f0[species_r=='co2'])[0]
    lpmax,lRp,lA0 = np.log10(pmax),np.log10(Rp),np.log10(A0)
    names  = [r"$\log\,$"+r"$p_{0}$",r"$\log\,$"+r"$f_{\rm O2}$",r"$\log\,$"+r"$f_{\rm H2O}$",r"$\log\,$"+r"$f_{\rm CO2}$",r"$\log\,$"+r"$R_{\rm p}$",r"$\log\,$"+r"$A_{\rm s}$"]
    truths = [lpmax,lfO2,lfH2O,lfCO2,lRp,lA0]
    ndim   = len(names)

    # import chain data
    samples,lnprob = reademceeh5(dirout+fnr+'.h5',nburn,thin)

    # print reduced chi-squared
    # lnp_max = np.amax(lnprob)
    # pos_max = np.where(lnprob == lnp_max)
    # print("Reduced chi-squared: ",-2*lnp_max/(dat.shape[0]-ndim))

    # relevant sizes
    nstep    = samples.shape[0]
    nwalkers = samples.shape[1]
    ndim     = samples.shape[2]

    return nburn, thin, names, ndim, truths

def plot_covariance_matrix(save_string, file_path='covariance_matrix.txt'):
  matrix = np.loadtxt(file_path, delimiter="\t")

  plt.imshow(matrix)
  plt.savefig(os.path.join('matrices',save_string))
  plt.close()

def true_vs_inferred(snr_uncorr, snr_tot, length, num_random, param_index):
  snr_uncorr_str = "{:.2f}".format(snr_uncorr)
  snr_tot_str = "{:.2f}".format(snr_tot)
  all_samples_uncorr = []
  all_samples_corr = []
  nburn, thin, names, ndim, truths = return_names_truths_etc()
  for i in range(num_random):
    samples_corr, _ = reademceeh5('outputs_correlated_052325/snr_uncorr_'+snr_uncorr_str+'_snr_tot_'+snr_tot_str+'_length_'+length+'/'+str(i+1)+'/earth_refl_demo_snr20_14dim.h5',nburn,thin)
    all_samples_corr.append(samples_corr.reshape((-1,ndim)))
    samples_uncorr, _ = reademceeh5('outputs_uncorrelated/snr_'+str(snr_tot)+'_'+str(i+1)+'/earth_refl_demo_snr20_14dim.h5',nburn,thin)
    all_samples_uncorr.append(samples_uncorr.reshape((-1,ndim)))

  all_samples_uncorr = np.array(all_samples_uncorr)
  all_samples_corr = np.array(all_samples_corr)
  avg_uncorr = np.mean(all_samples_uncorr, axis=0)
  avg_corr = np.mean(all_samples_corr, axis=0)

  median_uncorr = np.percentile(avg_uncorr[:, param_index], 50)
  median_corr = np.percentile(avg_corr[:, param_index], 50)

  lo_uncorr, hi_uncorr = np.percentile(avg_uncorr[:, param_index], [16, 84])
  lo_corr, hi_corr = np.percentile(avg_corr[:, param_index], [16, 84])

  param_name = names[param_index]
  title_str = r"{}: {:.2f} $\pm^{{+{:.2f}}}_{{-{:.2f}}}$".format(
        param_name, median_uncorr, hi_uncorr - median_uncorr, median_uncorr - lo_uncorr
        )+"\n"+r"{}: {:.2f} $\pm^{{+{:.2f}}}_{{-{:.2f}}}$".format(
        param_name, median_corr, hi_corr - median_corr, median_corr - lo_corr)   

  return title_str, avg_corr.T[param_index], avg_uncorr.T[param_index]

def plot_2d_hist(num_random, snr_uncorr, snr_tot, length):
  snr_uncorr_str = "{:.2f}".format(snr_uncorr)
  snr_tot_str = "{:.2f}".format(snr_tot)
  nburn, thin, names, ndim, truths = return_names_truths_etc()
  cfig, axes = plt.subplots(3, 2, figsize=(10,9))

  for i, ax in enumerate(axes.flat):
    title, corr_samples, uncorr_samples = true_vs_inferred(snr_uncorr, snr_tot, length, num_random, i=1) # i=1 for oxygen
    hist = corner.marginal(uncorr_samples, ax=ax, color='xkcd:lavender')
    ax.axvline(truths[i], lw=2, color='black')
    median_uncorr = np.percentile(uncorr_samples[i], 50)
    median_corr = np.percentile(corr_samples[i], 50)
    lo_uncorr, hi_uncorr = np.percentile(uncorr_samples[i], [16, 84])
    lo_corr, hi_corr = np.percentile(corr_samples[i], [16, 84])
    param_name = names[i]
    title_str = r"{}: {:.2f} $\pm^{{+{:.2f}}}_{{-{:.2f}}}$".format(
          param_name, median_uncorr, hi_uncorr - median_uncorr, median_uncorr - lo_uncorr
          )+"\n"+r"{}: {:.2f} $\pm^{{+{:.2f}}}_{{-{:.2f}}}$".format(
          param_name, median_corr, hi_corr - median_corr, median_corr - lo_corr)   
    ax.set_title(title_str, fontsize=8)
  plt.subplots_adjust( hspace=0.8)
  cfig.suptitle('SNR '+snr_tot_str+', Uncorrelated (black) vs Correlated (blue), '+str(num_random)+' noise instances', fontsize=18)
  cfig.savefig(os.path.join('plots/abundances','marginal_histograms_snr_14'))
  plt.close(cfig)

def plot_true_vs_inferred(num_random, snr_uncorr, snr_tot, length, snr_tot_list):
  snr_uncorr = 28.28
  cfig = plt.figure()
  for snr in snr_tot_list:
    title, corr_samples, uncorr_samples = true_vs_inferred(snr_uncorr, snr, length, num_random, i)
    median_uncorr = np.percentile(uncorr_samples[i], 50)
    median_corr = np.percentile(corr_samples[i], 50)
    lo_uncorr, hi_uncorr = np.percentile(uncorr_samples[i], [16, 84])
    lo_corr, hi_corr = np.percentile(corr_samples[i], [16, 84])
    plt.xerr(median_uncorr, )
  cfig.suptitle('SNR '+snr_tot_str+', Uncorrelated (black) vs Correlated (blue), '+str(num_random)+' noise instances', fontsize=18)
  cfig.savefig(os.path.join('plots/abundances','true_vs_inferred'))
  plt.close(cfig)
