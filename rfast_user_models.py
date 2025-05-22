import numpy        as     np
from scipy          import interpolate
from astropy.io     import ascii
from rfast_routines import readdat
#
# user-defined thermal structure model
#
# user note: must output temperature profile with Nlev elements (t) and a 
#            lower-boundary temperature (t0)
#
# inputs:
#
#       tpars   - parameters for thermal structure model, must agree w/rfast_inputs.scr
#           p   - pressure profile (Pa) [Nlev]
#        Ngas   - number of potential radiatively active gases, from set_gas_info
#       gasid   - gas names, from set_gas_info
#           f   - gas mixing ratio profiles [Ngas x Nlev]
#          fb   - background gas mixing ratio profile [Nlev]
#         mmr   - if true, mixing ratios are by mass. by volume, otherwise.
#           m   - atmospheric mean molecular weight profile (kg molecule**-1) [Nlev]
#          mb   - background gas molar weight (g mole**-1)
#
def atm_temp(tpars,p,Ngas,gasid,f,fb,mmr,m,mb):

  ########################
  ### user edits below ###

  # unpack user-defined temperature parameters; must agree with rfast_inputs
  tiso = tpars

  # number of atmospheric levels
  Nlev = p.shape[0]

  # set thermal structure profile; currently isothermal model
  t     = np.zeros(Nlev)
  t[:]  = tiso
  t0    = tiso

  ### user edits above ###
  ########################

  return t,t0

#
# user-defined surface albedo model
#
# user note: must output vector with length Nlam of lower-boundary albedos
#
# inputs:
#
#       Apars   - parameters for surface albedo model, must agree w/rfast_inputs.scr
#         lam   - wavelength array (um)
#
def surfalb(Apars,lam):

  ########################
  ### user edits below ###

  # unpack user-defined albedo parameters; must agree with rfast_inputs
  A0 = Apars

  # number of wavelength points
  Nlam  = lam.shape[0]

  # set surface flux albedo spectrum; currently grey
  As    = np.zeros(Nlam)
  As[:] = A0

  ### user edits above ###
  ########################

  return As

#
#
# user-defined cloud optical properties model
#
# user note: must output vectors with length Nlam. output must include gc=g1,g2,g3 as the 
#            first three moments of the phase function, wc as the single scattering albedo, 
#            and Qc as the extinction efficiency. scattering moments 2+ are only used in 
#            the phase angle-dependent reflected-light forward model.
#
# inputs:
#
#     cpars   -  parameters for cloud optical properties model, must agree w/rfast_inputs.scr
#       cld   -  boolean indicating if clouds are included; false -> zeroes out outputs
#     opdir   -  directory where hi-res opacities are located (string)
#       lam   -  wavelength array for output properties (um)
#
# outputs:
#
#     gc      -   cloud asymmetry parameter, moments 1--3 (len(3 x lam))
#     wc      -   cloud single scattering albedo (len(lam))
#     Qc      -   cloud extinction efficiency
#
def cloud_optprops(opars,cld,opdir,lam):

  # zero-out outputs
  gc1  = np.zeros(len(lam))
  gc2  = np.zeros(len(lam))
  gc3  = np.zeros(len(lam))
  wc   = np.zeros(len(lam))
  Qc   = np.zeros(len(lam))

  # if doing clouds
  if cld:

    ########################
    ### user edits below ###

    # example grey three-moment cloud setup

    # unpack user-defined cloud optical propert parameters; must agree with rfast_inputs
    #opars = w,g1,g2,g3
    #gc1 = np.zeros(len(lam))
    #gc2 = np.zeros(len(lam))
    #gc3 = np.zeros(len(lam))
    #wc  = np.zeros(len(lam))
    #Qc  = np.zeros(len(lam))
    #gc1[:] = g1
    #gc2[:] = g2
    #gc3[:] = g3
    #wc[:]  = w
    #Qc[:]  = 1

    # example 50/50 blend of Earth-like water liquid and ice clouds

    # liquid
    data     = readdat(opdir+'strato_cum.mie',19)
    lam_in   = data[:,0]
    w_in     = data[:,9]
    g_in     = data[:,10]
    q_in     = data[:,6]
    w_interp = interpolate.interp1d(lam_in,w_in,assume_sorted=True,fill_value="extrapolate")
    g_interp = interpolate.interp1d(lam_in,g_in,assume_sorted=True,fill_value="extrapolate")
    q_interp = interpolate.interp1d(lam_in,q_in,assume_sorted=True,fill_value="extrapolate")
    wcl      = w_interp(lam)
    gcl      = g_interp(lam)
    qcl      = q_interp(lam)
    # ice
    data     = readdat(opdir+'baum_cirrus_de100.mie',1)
    lam_in   = data[:,0]
    w_in     = data[:,1]
    g_in     = data[:,2]
    q_in     = data[:,3]
    w_interp = interpolate.interp1d(lam_in,w_in,assume_sorted=True,fill_value="extrapolate")
    g_interp = interpolate.interp1d(lam_in,g_in,assume_sorted=True,fill_value="extrapolate")
    q_interp = interpolate.interp1d(lam_in,q_in,assume_sorted=True,fill_value="extrapolate")
    wci      = w_interp(lam)
    gci      = g_interp(lam)
    qci      = q_interp(lam)
    # 50/50 mixture
    f      = 0.5
    wc     = f*wcl + (1-f)*wci
    Qc     = f*qcl + (1-f)*qci
    gc     = np.zeros(len(lam))
    gc1    = f*gcl + (1-f)*gci
    gc2    = np.zeros(len(lam))
    gc2[:] = 0.79 # only used if src = 'phas'
    gc3    = np.zeros(len(lam))
    gc3[:] = 0.67 # only used if src = 'phas'

    ### user edits above ###
    ########################

  # package together first and second moments
  gc = gc1,gc2,gc3

  return gc,wc,Qc
#
# user-defined cloud vertical structure model
#
# user note: must output cloud vertical differential extinction optical depths array 
#            of size Nlay (dtauc0). as this routine might alter the pressure grid to 
#            align with the cloud structure, it also returns the atmospheric structure 
#            interpolated onto this new pressure grid,
#
# inputs:
#
#       cpars   - parameters for cloud/haze/aerosol structure model, must agree w/rfast_inputs.scr
#         cld   -  boolean indicating if clouds are included; false -> zeroes out outputs
#           p   - pressure profile (Pa) [Nlev]
#           t   - temperature profile (K) [Nlev]
#           z   - altitude profile (m) [Nlev]
#        grav   - acceleration due to gravity profile (m s**-2) [Nlev]
#           f   - gas mixing ratio profiles [Ngas x Nlev]
#          fb   - background gas mixing ratio profile [Nlev]
#           m   - atmospheric mean molecular weight profile (kg molecule**-1) [Nlev]
# outputs:
#
#      dtauc0   - profile of layer aerosol differential extinction optical depths [Nlay]
#         atm   - p,t,z,grav,f,fb,m interpolated onto new pressure grid, if grid changes
#
def cloud_struct(cpars,cld,p,t,z,grav,f,fb,m):

  # number of atmospheric levels, layers
  Nlev = p.shape[0]
  Nlay = Nlev-1

  # cloud layer differential optical depths
  dtauc0 = np.zeros(Nlay)

  # store input pressure to later check for changes
  p0 = np.copy(p)

  # if doing a cloudy calc
  if cld:

    ########################
    ### user edits below ###

    # unpack user-defined cloud vertical structure parameters
    pt,dpc,tauc0 = cpars

    # force layers nearest to top/bottom of cloud to coincide with cloud
    it    = np.where(abs(p-pt) == min(abs(p-pt)))[0]
    if (it == Nlay):
      it = Nlay-1
    p[it] = pt
    pb    = pt + dpc
    ib    = np.where(abs(p-pb) == min(abs(p-pb)))[0]
    if (it == ib):
      ib  = it + 1
    p[ib] = pb
    dp    = p[1:] - p[:-1] # pressure difference across each layer

    # pressures encompassing cloud
    ip = np.argwhere( (p < pt+dpc) & (p >= pt) )

    # logic for cloud uniformly distributed in pressure
    dtauc0[ip[np.where(ip < Nlay)]] = dp[ip[np.where(ip < Nlay)]]/dpc*tauc0

    ### user edits above ###
    ########################

  # check if pressure grid changed, and interpolate structure onto new grid if so
  if( not (p0==p).all()):
    t_interp   = interpolate.interp1d(p0,t,assume_sorted=True,fill_value="extrapolate")
    t          = t_interp(p)
    f_interp   = interpolate.interp1d(p0,f,axis=1,fill_value="extrapolate")
    f          = f_interp(p)
    ft = np.sum(f,axis=0)
    fb         = 1 - ft
    fb[fb < 0] = 0
    m_interp   = interpolate.interp1d(p0,m,assume_sorted=True,fill_value="extrapolate")
    m          = m_interp(p)
    z_interp   = interpolate.interp1d(np.log10(p0),z,assume_sorted=True,fill_value="extrapolate")
    z          = z_interp(np.log10(p))
    g_interp   = interpolate.interp1d(np.log10(p0),grav,assume_sorted=True,fill_value="extrapolate")
    grav       = g_interp(np.log10(p))

  # return atmospheric structure, in case updated
  atm = p,t,z,grav,f,fb,m

  return dtauc0,atm