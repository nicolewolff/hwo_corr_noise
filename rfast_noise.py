# import statements
import time
import shutil
import sys
import numpy             as np
import matplotlib.pyplot as plt
from astropy.table       import Table, Column, MaskedColumn
from astropy.io          import ascii
from rfast_routines      import noise
from rfast_routines      import inputs
from scipy.ndimage import gaussian_filter
import george
from george import kernels

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

# input data filename
fn_dat = fns + '.raw'

# read input data
data        = ascii.read(dirout+fn_dat,data_start=1,delimiter='|')
lam         = data['col2'][:]
dlam        = data['col3'][:]
F1          = data['col4'][:]
F2          = data['col5'][:]

# snr0 constant w/wavelength case
if( len(snr0) == 1 ):
  if (ntype != 'cppm'):
    err = noise(lam0,snr0,lam,dlam,F2,Ts,ntype)
  else:
    err    = np.zeros(F2.shape[0])
    err[:] = 1/snr0
else: # otherwise snr0 is bandpass dependent
  err = np.zeros(len(lam))
  for i in range(0,len(snr0)):
    ilam = np.where(np.logical_and(lam >= lams[i], lam <= laml[i]))
    if (len(lam0) == 1): # lam0 may be bandpass dependent
      lam0i = lam0
    else:
      lam0i = lam0[i]
    if (ntype != 'cppm'):
      erri      = noise(lam0i,snr0[i],lam,dlam,F2,Ts,ntype)
      err[ilam] = erri[ilam]
    else:
      err[ilam] = 1/snr0[i]

# EDIT
def gen_gp(wl_channel, amp, length, uncorr_noise):    
    squared_exp_kernel = amp * george.kernels.ExpSquaredKernel(length)  # Removing amplitude squared: I think this is right?! just amplitude is my error bar, not squared...
    squared_exp_gp = george.GP(kernel=squared_exp_kernel)
    squared_exp_gp.compute(wl_channel, 0)  # error_bar "added in quadrature" to diagonal of covariance matrix doesn't work!
    return squared_exp_gp

# generate faux spectrum, with random noise if requested

data = np.copy(F2)

err_scaled = err * 1e10 # scaling up
F2_scaled = F2 * 1e10 # scaling up

if correlated:
  correlated_error = np.copy(F2)
  if not rnd:
    uncorr_noise = 0
  else:
    uncorr_noise = err[0]

  uv = np.array(lam[(lam >= lams[0]) & (lam <= lams[1])])
  vis = np.array(lam[(lam > lams[1]) & (lam <= lams[2])])
  nir = np.array(lam[lam > lams[2]])

  y_noise_uv = np.copy(uv)
  y_noise_vis = np.copy(vis)
  y_noise_nir = np.copy(nir)

  # UV: 
  flux_uv = np.array(F2_scaled[(lam >= lams[0]) & (lam <= lams[1])])
  for k in range(0,len(uv)):
    y_noise_uv[k]  = np.random.normal(flux_uv[k], err_scaled[k], 1)

  num_measurements = len(uv)
  gp = gen_gp(uv, arho, sigmarho, uncorr_noise * 1e10)
  y_sqexp = gp.sample(num_measurements)
  correlated_error[0:6] = y_sqexp / 1e10  # scaling down
  y_correlated_uv = y_noise_uv + y_sqexp
  # If < 100% correlated: make sure the amplitudes are right for everything

  # Vis:
  flux_vis = np.array(F2_scaled[(lam > lams[1]) & (lam <= lams[2])])
  num_measurements = len(vis)
  
  for k in range(0,len(vis)):
    y_noise_vis[k]  = np.random.normal(flux_vis[k], err_scaled[k], 1)

  gp = gen_gp(vis, arho, sigmarho, uncorr_noise * 1e10)
  y_sqexp = gp.sample(num_measurements)
  correlated_error[(lam > lams[1]) & (lam <= lams[2])] = y_sqexp / 1e10  # scaling down
  y_correlated_vis = y_noise_vis + y_sqexp

  # NIR: 
  flux_nir = np.array(F2_scaled[(lam > lams[2]) ])

  for k in range(0,len(nir)):
    y_noise_nir[k]  = np.random.normal(flux_nir[k], err_scaled[k], 1)  # ! doesn't work if err varies w/ wavelength
  num_measurements = len(nir)
  gp = gen_gp(nir, arho, sigmarho, uncorr_noise * 1e10)
  y_sqexp = gp.sample(num_measurements)
  correlated_error[(lam > lams[2]) ] = y_sqexp / 1e10
  y_correlated_nir = y_noise_nir + y_sqexp

  # Storing data:
  data[(lam >= lams[0]) & (lam <= lams[1])] = y_correlated_uv / 1e10
  data[(lam > lams[1]) & (lam <= lams[2])] = y_correlated_vis / 1e10
  data[(lam > lams[2]) ] = y_correlated_nir / 1e10
  
  if inclcovar:
    gp_uv = gen_gp(uv, arho, sigmarho, uncorr_noise)
    K_uv = gp_uv.kernel.get_value(uv[:, None])
    K_uv /= 1e20
    K_uv += np.diag(np.full(uv.shape, (err[0] ** 2)))

    gp_vis = gen_gp(vis, arho, sigmarho, uncorr_noise)
    K_vis = gp_vis.kernel.get_value(vis[:, None])
    K_vis /= 1e20
    K_vis += np.diag(np.full(vis.shape, (err[0]**2)))

    gp_nir = gen_gp(nir, arho, sigmarho, uncorr_noise)
    K_nir = gp_nir.kernel.get_value(nir[:, None])
    K_nir /= 1e20
    K_nir += np.diag(np.full(nir.shape, (err[0] ** 2)))

    cov_matrix = np.zeros((len(lam), len(lam)))
    cov_matrix[:len(uv), :len(uv)] = K_uv
    cov_matrix[len(uv):len(uv)+len(vis), len(uv):len(uv)+len(vis)] = K_vis
    cov_matrix[len(uv)+len(vis):, len(uv)+len(vis):] = K_nir

    np.savetxt("covariance_matrix.txt", cov_matrix, fmt="%.9e", delimiter="\t", header="Covariance Matrix")
    # np.savetxt("uv_matrix.txt", cov_matrix_uv, fmt="%.9e", delimiter="\t", header="UV Matrix")
    # np.savetxt("vis_matrix.txt", cov_matrix_vis, fmt="%.9e", delimiter="\t", header="VIS Matrix")
    # np.savetxt("nir_matrix.txt", cov_matrix_nir, fmt="%.9e", delimiter="\t", header="NIR Matrix")

  if (src == 'diff' or src == 'cmbn'):
    names = ['wavelength (um)','d wavelength (um)','albedo','flux ratio','data','uncertainty','correlated_error']
  if (src == 'thrm'):
    names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux (W/m**2/um)','data','uncertainty','correlated_error']
  if (src == 'scnd'):
    names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux ratio','data','uncertainty','correlated_error']
  if (src == 'trns'):
    names = ['wavelength (um)','d wavelength (um)','zeff (m)','transit depth','data','uncertainty','correlated_error']
  if (src == 'phas'):
    names = ['wavelength (um)','d wavelength (um)','reflect','flux ratio','data','uncertainty']
  data_out = Table([lam,dlam,F1,F2,data,err,correlated_error], names=names)

elif rnd:
  for k in range(0,len(lam)):
    data[k]  = np.random.normal(F2[k], err[k], 1)  # 'data' is the actual new randomized data point, somewhere within flux ratio +/- stddev
    if data[k] < 0:
      data[k] = 0.

  if (src == 'diff' or src == 'cmbn'):
    names = ['wavelength (um)','d wavelength (um)','albedo','flux ratio','data','uncertainty']
  if (src == 'thrm'):
    names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux (W/m**2/um)','data','uncertainty']
  if (src == 'scnd'):
    names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux ratio','data','uncertainty']
  if (src == 'trns'):
    names = ['wavelength (um)','d wavelength (um)','zeff (m)','transit depth','data','uncertainty']
  if (src == 'phas'):
    names = ['wavelength (um)','d wavelength (um)','reflect','flux ratio','data','uncertainty']
  data_out = Table([lam,dlam,F1,F2,data,err], names=names)
else:
  if (src == 'diff' or src == 'cmbn'):
    names = ['wavelength (um)','d wavelength (um)','albedo','flux ratio','data','uncertainty']
  if (src == 'thrm'):
    names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux (W/m**2/um)','data','uncertainty']
  if (src == 'scnd'):
    names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux ratio','data','uncertainty']
  if (src == 'trns'):
    names = ['wavelength (um)','d wavelength (um)','zeff (m)','transit depth','data','uncertainty']
  if (src == 'phas'):
    names = ['wavelength (um)','d wavelength (um)','reflect','flux ratio','data','uncertainty']
  data_out = Table([lam,dlam,F1,F2,data,err], names=names)

# END EDIT

# write data file
ascii.write(data_out,dirout+fnn+'.dat',format='fixed_width',overwrite=True)

# document parameters to file
shutil.copy(filename_scr,dirout+fnn+'.log')

# plot faux data
if (src == 'diff' or src == 'scnd' or src == 'cmbn' or src == 'phas'):
  ylab = 'Planet-to-Star Flux Ratio'
if (src == 'thrm'):
  ylab = r'Specific flux (W/m$^2$/${\rm \mu}$m)'
if (src == 'trns'):
  ylab = r'Transit depth'
plt.errorbar(lam, data, yerr=err, fmt=".k")
plt.ylabel(ylab)
plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
plt.savefig(dirout+fnn+'.png',format='png',bbox_inches='tight')
plt.close()
