# import os and manage threading
import os
os.environ["OMP_NUM_THREADS"] = "1" # recommended to prevent interference with emcee parallelization

# other import statements
import emcee
import time
import sys
import shutil
import scipy
import numpy             as np
import matplotlib.pyplot as plt
from multiprocessing     import Pool
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

# get input script filename
if len(sys.argv) >= 2:
  filename_scr = sys.argv[1] # if script name provided at command line
else:
  filename_scr = input("rfast inputs script filename: ") # otherwise ask for filename

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

covar = np.loadtxt("covariance_matrix.txt", delimiter="\t")
covar_inv = np.linalg.inv(covar)
np.savetxt("inverse_covariance_matrix.txt", covar_inv, fmt="%.9e", delimiter="\t", header="Inverse Covariance Matrix")

# unpackage parameters from user-defined routines
tiso = tpars[0]
A0 = Apars[0]
# no parameters for cloud optical properties model
pt,dpc,tauc0 = cpars

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
dat_upscaled = dat * 1e6
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
mode           = np.copy(lam)
mode[:]        = 1

# initialize disk integration quantities
threeD   = init_3d(src,ntg)

# initialize opacities and convolution kernels
sigma_interp,cia_interp,ncia,ciaid,kern = init(lam,dlam,lam_hr,species_l,species_c,opdir,pf,tf,mode=mode)

# initialize atmospheric model
p,t,t0,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
                                    tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                    species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                    mmri,mb,Mp,Rp,p10,fp10,src,ref,nu0)

# surface albedo model
As = surfalb(Apars,lam_hr)

# cloud optical properties: asymmetry parameter, single scattering albedo, extinction efficiency
gc,wc,Qc = cloud_optprops(opars,cld,opdir,lam_hr)

# cloud vertical structure model
dtauc0,atm = cloud_struct(cpars,cld,p,t,z,grav,f,fb,m)
p,t,z,grav,f,fb,m = atm

# min and max center-log ratio, if doing clr retrieval
if clr:
  n     = len(f0) + 1
  ximin = (n-1.)/n*(np.log(fmin) - np.log((1.-fmin)/(n-1.)))
  ximax = (n-1)/n*(np.log(1-n*fmin) - np.log(fmin))

# log-prior function
def lnprior(x):
  lpmax,lfO2,lfH2O,lfCO2,lRp,lA0 = x
  pmax,fO2,fH2O,fCO2,Rp,A0 = 10**(lpmax),10**(lfO2),10**(lfH2O),10**(lfCO2),10**(lRp),10**(lA0)
  f0[species_r=='o2'],f0[species_r=='h2o'],f0[species_r=='co2'] = fO2,fH2O,fCO2

  # sum gaussian priors
  lng = 0.0 

  # prior limits
  if 1.0 <= pmax <= 100000000.0 and 1e-10 <= fO2 <= 1.0 and 1e-10 <= fH2O <= 1.0 and 1e-10 <= fCO2 <= 1.0 and 0.1 <= Rp <= 10.0 and 0.01 <= A0 <= 1.0 and np.sum(f0) <= 1 and Rp > 0 and Mp > 0 and pt + dpc < pmax:
    return 0.0 + lng
  return -np.inf

# log-likelihood function
def lnlike(x):

  # reverts to using Mp if gp is not retrieved
  gp = -1

  lpmax,lfO2,lfH2O,lfCO2,lRp,lA0 = x
  # not performing clr retrieval
  pmax,fO2,fH2O,fCO2,Rp,A0 = 10**(lpmax),10**(lfO2),10**(lfH2O),10**(lfCO2),10**(lRp),10**(lA0)
  f0[species_r=='o2'],f0[species_r=='h2o'],f0[species_r=='co2'] = fO2,fH2O,fCO2

  # package parameters for user-defined routines
  tpars = tiso
  Apars = A0
  # no parameters for cloud optical properties model
  cpars = pt,dpc,tauc0

  # package parameters for call to forward model
  x0 = f0,pmax,Rp,Mp,gp,Apars,tpars,opars,cpars,fc,a,alpha,mb,rayb
  y  = lam_hr,pmin,Ngas,mmw0,nu0,em,Ts,Rs,ray0,gasid,ncia,ciaid,species_l,species_c,\
       Rpi,sigma_interp,cia_interp,kern,ray,ref,cld,sct,phfc,fixp,pf,fixt,tf,mmrr,\
       p10,fp10,src,threeD,fntmp,skptmp,colt,colpt,psclt,species_r,fnatm,skpatm,\
       colr,colpr,psclr,Nlev
  if inclcovar:
    total_ll = -0.5 * ((dat - Fx(x0,y)).T @ covar_inv @ (dat - Fx(x0,y))) 
    return total_ll
  else:
    return -0.5*np.sum((dat-Fx(x0,y))**2/err**2) 

# log-probability from Bayes theorem
def lnprob(x):
  lp = lnprior(x)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(x)

# forward model for emcee and analysis purposes; re-packages gen_spec routine
def Fx(x,y):

  f0,pmax,Rp,Mp,gp,Apars,tpars,opars,cpars,fc,a,alpha,mb,rayb = x
  lam,pmin,Ngas,mmw0,nu0,em,Ts,Rs,ray0,gasid,ncia,ciaid,species_l,species_c,\
  Rpi,sigma_interp,cia_interp,kern,ray,ref,cld,sct,phfc,fixp,pf,fixt,tf,mmrr,\
  p10,fp10,src,threeD,fntmp,skptmp,colt,colpt,psclt,species_r,fnatm,skpatm,\
  colr,colpr,psclr,Nlev = y

  # do not read in thermal structure
  rdtmp = False

  # do not read in atmospheric structure
  rdgas = False

  # initialize atmospheric model
  p,t,t0,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
                                      tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                      species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                      mmrr,mb,Mp,Rp,p10,fp10,src,ref,nu0,gp=gp)

  # surface albedo model
  As = surfalb(Apars,lam)

  # cloud optical properties: asymmetry parameter, single scattering albedo, extinction efficiency
  gc,wc,Qc = cloud_optprops(opars,cld,opdir,lam)

  # cloud vertical structure model
  dtauc0,atm = cloud_struct(cpars,cld,p,t,z,grav,f,fb,m)
  p,t,z,grav,f,fb,m = atm

  # call forward model
  F1,F2 = gen_spec(Nlev,Rp,a,As,em,p,t,t0,m,z,grav,Ts,Rs,ray,ray0,rayb,f,fb,
                   mb,mmw0,mmrr,ref,nu,alpha,threeD,
                   gasid,ncia,ciaid,species_l,species_c,
                   cld,sct,phfc,fc,gc,wc,Qc,dtauc0,lamc0,
                   src,sigma_interp,cia_interp,lam,pf=pf,tf=tf)

  # degrade resolution
  F_out = kernel_convol(kern,F2)

  # "distance" scaling for thermal emission case
  if (src == 'thrm'):
    F_out = ( F_out*(Rp/Rpi)**2 )

  return F_out

# syntax to identify main core of program
if __name__ == '__main__':

  # inform user of key opacities information
  opacities_info(opdir)

  # test forward model
  x  = f0,pmax,Rp,Mp,gp,Apars,tpars,opars,cpars,fc,a,alpha,mb,rayb
  y  = lam_hr,pmin,Ngas,mmw0,nu0,em,Ts,Rs,ray0,gasid,ncia,ciaid,species_l,species_c,\
       Rpi,sigma_interp,cia_interp,kern,ray,ref,cld,sct,phfc,fixp,pf,fixt,tf,mmrr,\
       p10,fp10,src,threeD,fntmp,skptmp,colt,colpt,psclt,species_r,fnatm,skpatm,\
       colr,colpr,psclr,Nlev
  if (src == 'diff' or src == 'scnd' or src == 'cmbn' or src == 'phas'):
    ylab = 'Planet-to-Star Flux Ratio'
  if (src == 'thrm'):
    ylab = r'Specific flux (W/m$^2$/${\rm \mu}$m)'
  if (src == 'trns'):
    ylab = r'Transit depth'
  plt.errorbar(lam, dat, yerr=err, fmt=".k")
  plt.plot(lam,Fx(x,y), drawstyle='steps-mid')
  plt.ylabel(ylab)
  plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
  plt.savefig(dirout+fnr+'_test.png',format='png',bbox_inches='tight')
  plt.close()

  # document parameters to file
  shutil.copy(filename_scr,dirout+fnr+'.log')

  # g(x) after benneke & seager (2012); only needed if doing clr retrieval
  if clr:
    gx = np.exp((np.sum(np.log(f0)) + np.log(max(fmin,1-np.sum(f0))))/(len(f0) + 1))

  # unpackage parameters from user-defined routines
  tiso = tpars[0]
  A0 = Apars[0]
  # no parameters for cloud optical properties model
  pt,dpc,tauc0 = cpars

  # retrieved parameters initial guess
  lfO2,lfH2O,lfCO2 = np.log10(f0[species_r=='o2'])[0],np.log10(f0[species_r=='h2o'])[0],np.log10(f0[species_r=='co2'])[0]
  lpmax,lRp,lA0 = np.log10(pmax),np.log10(Rp),np.log10(A0)
  guess = [lpmax,lfO2,lfH2O,lfCO2,lRp,lA0]
  ndim  = len(guess)

  # create backup / save file; prevent h5 overwrite or check if restart h5 exists
  if not restart:
    if os.path.isfile(dirout+fnr+'.h5'):
      # print("rfast warning | major | h5 file already exists")
      quit()
    else:
      backend  = emcee.backends.HDFBackend(dirout+fnr+'.h5')
      backend.reset(nwalkers, ndim)
      # initialize walkers as a cloud around guess
      pos = guess + 1e-4*np.random.randn(nwalkers,ndim)
  else:
    if not os.path.isfile(dirout+fnr+'.h5'):
      # print("rfast warning | major | h5 does not exist for restart")
      quit()
    else:
      # otherwise initialize walkers from existing backend
      backend  = emcee.backends.HDFBackend(dirout+fnr+'.h5')
      pos = backend.get_last_sample()

  # timing
  tstart = time.time()

  # multiprocessing implementation
  with Pool(60) as pool: # empty paramter means using all cores, i believe 128

    # initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool)

    # run the mcmc
    sampler.run_mcmc(pos, nstep, progress=progress)

  # timing
  tend = time.time()

  output_folder = "timestamps"
  os.makedirs(output_folder, exist_ok=True)
  
  timing_filename = dirout.replace("/", "_") + ".txt"
  timing_filepath = os.path.join(output_folder, timing_filename)

  with open(timing_filepath, "a") as f:
      f.write(f"Retrieval timing (hr): {(tend-tstart) / 3600}\n")
  print('Retrieval timing (s): ',tend-tstart)
