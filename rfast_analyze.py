# import statements
import emcee
import corner
import h5py
import sys
import os
import numpy             as np
import matplotlib.pyplot as plt
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
  filename_scr = input("rfast inputs script filename: ") # otherwise ask for filename
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
lnp_max = np.amax(lnprob)
pos_max = np.where(lnprob == lnp_max)
print("Reduced chi-squared: ",-2*lnp_max/(dat.shape[0]-ndim))
output_folder = "chi_squared"
os.makedirs(output_folder, exist_ok=True)

chi_squared_filename = dirout.replace("/", "_") + ".txt"
chi_squared_filepath = os.path.join(output_folder, chi_squared_filename)

with open(chi_squared_filepath, "a") as f:
  f.write(f"Reduced chi-squared: {-2*lnp_max/(dat.shape[0]-ndim)}\n")

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

# plot the walker positions in each step
fig, axes = plt.subplots(ndim, 1, figsize=(8, 4 * ndim), tight_layout=True)
for i in range(ndim):
  for j in range(0,nwalkers):
    axes[i].plot(samples[:,j,i],color="black",linewidth=0.5)
    axes[i].set_ylabel(str(names[i]))
    axes[i].set_xlabel('Step')
plt.savefig(dirout+fnr+'_walkers.png',format='png')
plt.close()

# plotting the corner plot
fig = corner.corner(samples.reshape((-1,ndim)), quantiles=[0.16, 0.5, 0.84],show_titles=True,
                    color='xkcd:steel blue', labels=names, truths=truths, 
                    title_fmt = '.4f')
fig.savefig(dirout+fnr+'_corner.png',format='png',bbox_inches='tight')
plt.close(fig)

# plot best-fit model and residuals
gp = -1 # reverts to using Mp if gp not retrieved

# get best-fit parameters
lpmax,lfO2,lfH2O,lfCO2,lRp,lA0 = samples[pos_max][0]
pmax,fO2,fH2O,fCO2,Rp,A0 = 10**(lpmax),10**(lfO2),10**(lfH2O),10**(lfCO2),10**(lRp),10**(lA0)
f0[species_r=='o2'],f0[species_r=='h2o'],f0[species_r=='co2'] = fO2,fH2O,fCO2

# package parameters for user-defined routines
tpars = tiso
Apars = A0
# no parameters for cloud optical properties model
cpars = pt,dpc,tauc0

x0 = f0,pmax,Rp,Mp,gp,Apars,tpars,opars,cpars,fc,a,alpha,mb,rayb
y  = lam_hr,pmin,Ngas,mmw0,nu0,em,Ts,Rs,ray0,gasid,ncia,ciaid,species_l,species_c,\
     Rpi,sigma_interp,cia_interp,kern,ray,ref,cld,sct,phfc,fixp,pf,fixt,tf,mmrr,\
     p10,fp10,src,threeD,fntmp,skptmp,colt,colpt,psclt,species_r,fnatm,skpatm,\
     colr,colpr,psclr,Nlev

# determine correct label for y axis
if (src == 'diff' or src == 'scnd' or src == 'cmbn' or src == 'phas'):
  ylab = 'Planet-to-Star Flux Ratio'
if (src == 'thrm'):
  ylab = r'Specific flux (W/m$^2$/${\rm \mu}$m)'
if (src == 'trns'):
  ylab = r'Transit depth'

# best-fit model
from rfast_retrieve import Fx
plt.errorbar(lam, dat, yerr=err, fmt=".k")
plt.plot(lam, Fx(x0,y), drawstyle='steps-mid')
plt.ylabel(ylab)
plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
plt.savefig(dirout+fnr+'_bestfit.png',format='png',bbox_inches='tight')
plt.close()

# plotting all models
'''
for idx, sample in enumerate(samples):
  lpmax,lfO2,lfH2O,lfCO2,lRp,lA0 = sample[0]
  pmax,fO2,fH2O,fCO2,Rp,A0 = 10**(lpmax),10**(lfO2),10**(lfH2O),10**(lfCO2),10**(lRp),10**(lA0)
  f0[species_r=='o2'],f0[species_r=='h2o'],f0[species_r=='co2'] = fO2,fH2O,fCO2
  x0 = f0,pmax,Rp,Mp,gp,Apars,tpars,opars,cpars,fc,a,alpha,mb,rayb
  y  = lam_hr,pmin,Ngas,mmw0,nu0,em,Ts,Rs,ray0,gasid,ncia,ciaid,species_l,species_c,\
      Rpi,sigma_interp,cia_interp,kern,ray,ref,cld,sct,phfc,fixp,pf,fixt,tf,mmrr,\
      p10,fp10,src,threeD,fntmp,skptmp,colt,colpt,psclt,species_r,fnatm,skpatm,\
      colr,colpr,psclr,Nlev
  from rfast_retrieve import Fx
  plt.errorbar(lam, (dat), yerr=err, fmt=".k")
  plt.plot(lam, (Fx(x0,y)), drawstyle='steps-mid')
  plt.ylabel(ylab)
  plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
  plt.savefig(dirout+'model_'+str(idx)+'.png',format='png',bbox_inches='tight')
  plt.close()
'''

# residuals
plt.errorbar(lam, dat- Fx(x0,y), yerr=err, fmt=".k")
plt.ylabel(ylab)
plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
plt.savefig(dirout+fnr+'_residuals.png',format='png',bbox_inches='tight')
plt.close()

# compute & print parameters, truths, mean inferred, and 16/84 percentile (credit: arnaud)
mean = np.zeros(len(names))
std  = np.zeros([2,len(names)])
for i in range(len(names)):
  prcnt    = np.percentile(samples[:,:,i], [16, 50, 84])
  mean[i]  = prcnt[1]
  std[0,i] = np.diff(prcnt)[0]
  std[1,i] = np.diff(prcnt)[1]
colnames = ['Parameter','Input','Mean','- sig','+ sig']
data_out = Table([names,truths,mean,std[0,:],std[1,:]],names=colnames)
ascii.write(data_out,dirout+fnr+'.tab',format='fixed_width',overwrite=True)