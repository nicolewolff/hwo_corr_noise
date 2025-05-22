### HOW TO RUN: specify a directory where the main rfast_inputs.scr is found.
### manually fill in all the other directories with data to overplot
### eventually make this easier with a script

# import statements
import emcee
import corner
import h5py
import sys
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

# simple routine for importing emcee chain from h5 file
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

# get input script filename
if len(sys.argv) >= 2:
  filename_scr = sys.argv[1]    # if script name provided at command line
else:
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

# if doing center-log ratio, transform back to mixing ratio
if clr:
  gind = []
  xi   = samples[:,:,gind]
  clrs = np.sum(np.exp(xi),axis=2) + np.exp(-np.sum(xi,axis=2))
  clrs = np.repeat(clrs[:,:,np.newaxis], len(gind), axis=2)
  samples[:,:,gind] = np.log10(np.divide(np.exp(samples[:,:,gind]),clrs))

# plot the corner plot

red_line = mlines.Line2D([], [], color='maroon', label='SNR 7.07')
purple_line = mlines.Line2D([], [], color='xkcd:pale purple', label='SNR 10')
green_line = mlines.Line2D([], [], color='forestgreen', label='SNR 14.14')
blue_line = mlines.Line2D([], [], color='deepskyblue', label='SNR 20')

def overlay_corner_plots(num_iterations, clr, snr, handle):
  snr_string = "{:.2f}".format(snr)
  samples, _ = reademceeh5('outputs_correlated_debugging/snr_14.14_0.5correlated_length_4e-3/1/earth_refl_demo_snr20_14dim.h5',nburn,thin)
  fig = corner.corner(samples.reshape((-1,ndim)), quantiles=[],show_titles=False, 
                      color='forestgreen', labels=names, truths=truths, plot_contours=False, fill_contours=False)

  # samples, _ = reademceeh5('outputs_correlated_debugging/snr_14.14_0.5correlated_length_4e-3/2/earth_refl_demo_snr20_14dim.h5',nburn,thin)
  # corner.corner(samples.reshape((-1,ndim)), quantiles=[],show_titles=False, 
  #                     color='xkcd:sea blue', labels=names, truths=truths, plot_contours=False, fill_contours=False, fig=fig)

  samples, _ = reademceeh5('outputs_uncorrelated/snr_'+snr_string+'_1/earth_refl_demo_snr20_14dim.h5',nburn,thin)
  corner.corner(samples.reshape((-1,ndim)), quantiles=[],show_titles=False, 
                      color='black', labels=names, truths=truths, plot_contours=False, fill_contours=False, fig=fig)


  medians = []
  medians_uncorr = []
  for i in range(num_iterations):
    samples, _ = reademceeh5('outputs_uncorrelated/snr_'+snr_string+'_'+str(i+1)+'/earth_refl_demo_snr20_14dim.h5',nburn,thin)
    median_uncorr = np.median(samples.reshape((-1,ndim)), axis=0)
    medians_uncorr.append(median_uncorr)
    
    samples_corr, _ = reademceeh5('outputs_correlated_debugging/snr_'+snr_string+'_0.5correlated_length_4e-3/'+str(i+1)+'/earth_refl_demo_snr20_14dim.h5',nburn,thin)
    # corner.corner(samples_corr.reshape((-1,ndim)), quantiles=[],
    #                 color=clr, truths=truths, plot_contours = False, fill_contours=False,
    #                 fig=fig)
    median = np.median(samples_corr.reshape((-1,ndim)), axis=0)
    medians.append(median)

  median_mean_uncorr = np.mean(medians_uncorr, axis=0)
  stddev = np.std(medians_uncorr, axis=0)
  corner.overplot_lines(fig, median_mean_uncorr, color="black")
  corner.overplot_lines(fig, median_mean_uncorr - stddev, color="black", linestyle='--')
  corner.overplot_lines(fig, median_mean_uncorr + stddev, color="black", linestyle='--') 

  median_mean = np.mean(medians, axis=0)
  stddev = np.std(medians, axis=0)
  corner.overplot_lines(fig, median_mean, color="xkcd:violet")
  corner.overplot_lines(fig, median_mean - stddev, color="xkcd:violet", linestyle='--')
  corner.overplot_lines(fig, median_mean + stddev, color="xkcd:violet", linestyle='--') 

  plt.legend(handles=[handle], bbox_to_anchor=(-2.15, 4.15, 1., .0), loc=1,fontsize='xx-large')
  fig.savefig('plots/snr_'+str(snr)+'_0.5correlated_vs_uncorrelated_4e-3_corner.png',format='png',bbox_inches='tight')
  plt.close(fig)


  samples, _ = reademceeh5('outputs_correlated_debugging/snr_14.14_0.5correlated_length_4e-3/1/earth_refl_demo_snr20_14dim.h5',nburn,thin)
  # plt.hist(samples.reshape((-1,ndim))[:,1], bins=30, color='xkcd:sea blue', histtype='step')
  plt.axvline(median_mean_uncorr[1], color='black')
  stddev = np.std(medians_uncorr, axis=0)
  plt.axvline(median_mean_uncorr[1] - stddev[1], color='black', linestyle='--')
  plt.axvline(median_mean_uncorr[1] + stddev[1], color='black', linestyle='--')

  plt.axvline(median_mean[1], color='xkcd:violet')
  stddev = np.std(medians, axis=0)
  plt.axvline(median_mean[1] - stddev[1], color='xkcd:violet', linestyle='--')
  plt.axvline(median_mean[1] + stddev[1], color='xkcd:violet', linestyle='--')

  samples, _ = reademceeh5('outputs_correlated_debugging/snr_14.14_0.5correlated_length_4e-3/2/earth_refl_demo_snr20_14dim.h5',nburn,thin)
  plt.hist(samples.reshape((-1,ndim))[:,1], bins=30, color='forestgreen', histtype='step', label='Correlated')

  samples, _ = reademceeh5('outputs_uncorrelated/snr_'+snr_string+'_1/earth_refl_demo_snr20_14dim.h5',nburn,thin)
  plt.hist(samples.reshape((-1,ndim))[:,1], bins=30, color='black', histtype='step', label='Uncorrelated')
  plt.title(r"$\mathrm{O}_2$ Abundance, SNR = 14")
  plt.legend(loc='upper left')
  plt.savefig('plots/snr_'+str(snr)+'_0.5correlated_vs_uncorrelated_4e-3_hist2d.png',format='png',bbox_inches='tight')

  plt.close()
  return 

overlay_corner_plots(5, 'forestgreen', 14.14, green_line)