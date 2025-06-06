# import statements
import sys
import os
import shutil
import numpy        as     np
from astropy.io     import ascii
from rfast_routines import inputs

# get input script filename
if len(sys.argv) >= 3:
  filename_scr = sys.argv[1] # if script name provided at command line
  filename_ret = sys.argv[2]
else:
  filename_scr   = input("rfast inputs script filename: ") # otherwise ask for filename
  filename_ret   = input("rfast retrieved parameters filename: ") # otherwise ask for filename

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

# uniform minimum mixing ratio for clr retrievals; after benneke & seager (2012)
if clr:
  print("rfast note | retrieval is center-log (ignores rpars lin/log) with uniform distribution above mixing ratio of: ",fmin)

# read non-additional prior lines in retrieved parameters file
npar = 0
with open(filename_ret,'r') as fp:
  # read all lines
  lines = fp.readlines()
  for row in lines:
    row = row.split("|")
    if (len(row) > 6):
      npar = npar + 1

# read input table
tab  = ascii.read(filename_ret,data_start=1,data_end=npar+1,delimiter='|')
par  = tab['col1']
lab  = tab['col2']
ret  = tab['col3']
log  = tab['col4']
shp  = tab['col5']
p1   = tab['col6']
p2   = tab['col7']

# number of read-in parameters
npar = par.shape[0]

# number of retrieved, logged, gas parameters, and user-defined model parameters
# check that retrieved gases are active; check for retrieved Mp and gp (a big no-no!)
nret  = 0
nlog  = 0
ngas  = 0
nlgas = 0
ntusr = 0
nAusr = 0
nousr = 0
ncusr = 0
mf    = False
gf    = False
for i in range(npar):
  if (ret[i].lower() == 'y'): # check if parameter is retrieved
    nret = nret + 1
    if (par[i] == 'Mp'): # flag if planet mass is retrieved
      mf  = True
    if (par[i] == 'gp'): # flag if surface gravity is retrieved
      gf  = True
    if (log[i].lower() == 'log'): # check if parameter is retrieved as log
      nlog = nlog + 1
    if (par[i][0] == 'f' and par[i] != 'fc'): # check if parameter is a gas abundance
      ngas = ngas + 1
      if (log[i].lower() == 'log'): # check if gas abundance retrieval is log
        nlgas = nlgas + 1
      if (len(species_r) <= 1):
        if not (species_r[0] == par[i][1:].lower()):
          print("rfast warning | major | requested retrieved gas is not radiatively active; ",par[i][1:].lower())
          quit()
      else:
        if not any(species_r == par[i][1:].lower()):
          print("rfast warning | major | requested retrieved gas is not radiatively active; ",par[i][1:].lower())
          quit()
    if (par[i][0:2] == 't:'): # count retrieved user-defined thermal structure parameters
      ntusr = ntusr + 1
    if (par[i][0:2] == 'A:'): # count retrieved user-defined surface albedo parameters
      nAusr = nAusr + 1
    if (par[i][0:2] == 'o:'): # count retrieved user-defined cloud optical properties parameters
      nousr = nousr + 1
    if (par[i][0:2] == 'c:'): # count retrieved user-defined cloud structure parameters
      ncusr = ncusr + 1

# warning if no parameters are retrieved
if (nret == 0):
  print("rfast warning | major | zero requested retrieved parameters")
  quit()

# warning that you cannot retrieve on both Mp and gp
if (mf and gf):
  print("rfast warning | major | cannot retrieve on both Mp and gp")
  quit()

# warning if clr retrieval is requested but no gases are retrieved
if (clr and ngas == 0):
  print("rfast warning | minor | center-log retrieval requested but no retrieved gases")

# warning that clr treatment assumes gases retrieved in log space
if (clr and ngas != nlgas):
  print("rfast warning | minor | requested center-log retrieval transforms all gas constraints to log-space")

# warning if clr retrieval and number of included gases is smaller than retrieved gases
if (clr and ngas < len(f0)):
  print("rfast warning | major | center-log retrieval functions only if len(f0) equals number of retrieved gases")
  quit()

# note number of retrieved parameters for user-defined models
if (ntusr > 0):
  print("rfast note | retrieved parameters in user-defined thermal structure model: ",ntusr)
if (nAusr > 0):
  print("rfast note | retrieved parameters in user-defined surface albedo model: ",nAusr)
if (nousr > 0):
  print("rfast note | retrieved parameters in user-defined cloud optical properties model: ",nousr)
if (ncusr > 0):
  print("rfast note | retrieved parameters in user-defined cloud structure model: ",ncusr)

# determine user-defined model parameters
targetut1 = "# no parameters for thermal structure model"
targetuA1 = "# no parameters for surface albedo model"
targetuo1 = "# no parameters for cloud optical properties model"
targetuc1 = "# no parameters for cloud structure model"
targetut2 = "# no parameters for thermal structure model"
targetuA2 = "# no parameters for surface albedo model"
targetuo2 = "# no parameters for cloud optical properties model"
targetuc2 = "# no parameters for cloud structure model"
with open(r'rfast_user_models.py', 'r') as fp:
  # read all lines
  lines = fp.readlines()
  for row in lines:
    rows = row.strip()
    if (len(rows) > 0):
      if (rows[0] != '#'):
        rowsr = rows.replace(" ","")
        if rowsr.find("=tpars") != -1:
          targetut1 = rows.split("#")[0]
          targetut2 = targetut1.split("#")[0].split("=")[1].replace(" ","") + " = " + targetut1.split("#")[0].split("=")[0].replace(" ","")
          Ntpars    = len(targetut1.split("#")[0].split("=")[0].split(","))
          if (Ntpars == 1):
            targetut1 = targetut1 + "[0]"
          if (ntusr > Ntpars):
            print("rfast warning | major | retrieved parameters in user-defined thermal structure model exceeds input")
            quit()
        if rowsr.find("=Apars") != -1:
          targetuA1 = rows
          targetuA2 = targetuA1.split("#")[0].split("=")[1].replace(" ","") + " = " + targetuA1.split("#")[0].split("=")[0].replace(" ","")
          NApars    = len(targetuA1.split("#")[0].split("=")[0].split(","))
          if (NApars == 1):
            targetuA1 = targetuA1 + "[0]"
          if (nAusr > NApars):
            print("rfast warning | major | retrieved parameters in user-defined surface albedo model exceeds input")
            quit()
        if rowsr.find("=opars") != -1:
          targetuo1 = rows
          targetuo2 = targetuo1.split("#")[0].split("=")[1].replace(" ","") + " = " + targetuo1.split("#")[0].split("=")[0].replace(" ","")
          Nopars    = len(targetuo1.split("#")[0].split("=")[0].split(","))
          if (Nopars == 1):
            targetuo1 = targetuo1 + "[0]"
          if (nousr > Nopars):
            print("rfast warning | major | retrieved parameters in user-defined cloud optical properties model exceeds input")
            quit()
        if rowsr.find("=cpars") != -1:
          targetuc1 = rows
          targetuc2 = targetuc1.split("#")[0].split("=")[1].replace(" ","") + " = " + targetuc1.split("#")[0].split("=")[0].replace(" ","")
          Ncpars    = len(targetuc1.split("#")[0].split("=")[0].split(","))
          if (Ncpars == 1):
            targetuc1 = targetuc1 + "[0]"
          if (ncusr > Ncpars):
            print("rfast warning | major | retrieved parameters in user-defined cloud structure model exceeds input")
            quit()

# get additional priors specified in retrieved parameters file
with open(filename_ret,'r') as fp:
  # read all lines
  lines = fp.readlines()
  for row in lines:
    if row.find("other priors") != -1:
      otherp = row.split(":")
      if (len(otherp) > 1):
        np  = len(otherp) - 1
        var = " and "
        for i in range(1,len(otherp)):
          var = var + otherp[i].strip()
          if (i < np):
            var = var + " and "
        otherp = var
      else:
        otherp = ""

# set up initial guess syntax
iret    = 0
igas    = 0
ilog    = 0
guessl1 = ""
guessl2 = ""
guessl3 = "guess = ["
guessl4 = ""
guessl5 = ""
priorl1 = ""
priorl2 = ""
for i in range(npar):
  if (ret[i].lower() == 'y'):
    if (log[i].lower() == 'lin'):
      if (par[i][0] == 'f' and par[i] != 'fc'):
        if (not clr):
          guessl1 = guessl1 + par[i]
          if (len(species_r) == 1):
            guessl2 = guessl2 + "f0[0]"
          else:
            guessl2 = guessl2 + "f0[species_r=='" + par[i][1:].lower() + "'][0]"
          guessl3 = guessl3 + par[i]
        else:
          guessl1 = guessl1 + 'xi' + par[i][1:]
          if (len(species_r) == 1):
            guessl2 = guessl2 + "np.log(f0[0]/gx)"
          else:
            guessl2 = guessl2 + "np.log(f0[species_r=='" + par[i][1:].lower() + "'][0]/gx)"
          guessl3 = guessl3 + "xi" + par[i][1:]
        igas = igas + 1
        if (igas < ngas):
          guessl1 = guessl1 + ","
          guessl2 = guessl2 + ","
      else:
        var = par[i]
        if par[i].find(":") != -1:
          var = par[i].split(":")[1]
        guessl3 = guessl3 + var
    if (log[i].lower() == 'log'):
      if (par[i][0] == 'f' and par[i] != 'fc'):
        if (not clr):
          guessl1 = guessl1 + "l" + par[i]
          if (len(species_r) == 1):
            guessl2 = guessl2 + "np.log10(f0[0])"
          else:
            guessl2 = guessl2 + "np.log10(f0[species_r=='" + par[i][1:].lower() + "'])[0]"
          guessl3 = guessl3 + "l" + par[i]
        else:
          guessl1 = guessl1 + 'xi' + par[i][1:]
          if (len(species_r) == 1):
            guessl2 = guessl2 + "np.log(f0[0]/gx)"
          else:
            guessl2 = guessl2 + "np.log(f0[species_r=='" + par[i][1:].lower() + "'][0]/gx)"
          guessl3 = guessl3 + "xi" + par[i][1:]
        igas = igas + 1
        if (igas < ngas):
          guessl1 = guessl1 + ","
          guessl2 = guessl2 + ","
      else:
        var = par[i]
        if par[i].find(":") != -1:
          var = par[i].split(":")[1]
        guessl3 = guessl3 + "l" + var
        guessl4 = guessl4 + "l" + var
        guessl5 = guessl5 + "np.log10(" + var + ")"
        priorl1 = priorl1 + var
        priorl2 = priorl2 + "10**(l" + var + ")"
        if (ilog < nlog-nlgas-1):
          guessl4 = guessl4 + ","
          guessl5 = guessl5 + ","
          priorl1 = priorl1 + ","
          priorl2 = priorl2 + ","
        ilog = ilog + 1     
    if (iret < nret-1):
      guessl3 = guessl3 + ","
    if (iret == nret-1):
      guessl3 = guessl3 + "]"
    iret = iret + 1
targetg1 = "# no log-retrieved gases or not center-log retrieval"
if (nlgas >= 1 or (clr and ngas > 0)):
  targetg1 = guessl1 + " = " + guessl2
targetg2 = "# no log-retrieved non-gas parameters"
if (nlog-nlgas >= 1):
  targetg2 = guessl4 + " = " + guessl5
targetg3 = guessl3

# set up likelihood syntax
iret   = 0
ilog   = 0
igas   = 0
likel1 = ""
likel2 = ""
likel3 = ""
likel4 = ""
likel5 = ""
likel6 = ""
for i in range(npar):
  if (ret[i].lower() == 'y'):
    if (par[i][0] == 'f' and par[i] != 'fc' and clr):
      likel1 = likel1 + "xi" + par[i][1:]
      likel2 = likel2 + par[i]
      likel3 = likel3 + "np.exp(xi" + par[i][1:] + ")/clrs"
      if (len(species_r) == 1):
        likel4 = likel4 + "f0[0]"
      else:
        likel4 = likel4 + "f0[species_r=='" + par[i][1:].lower() + "']"
      likel5 = likel5 + par[i]
      likel6 = likel6 + "xi" + par[i][1:]
      if (log[i].lower() == 'log'):
        ilog = ilog + 1
      if (igas < ngas and i < npar):
        likel2 = likel2 + ","
        likel3 = likel3 + ","
        likel6 = likel6 + ","
      igas = igas + 1
      if (igas < ngas):
        likel4 = likel4 + ","
        likel5 = likel5 + ","
    else:
      var = par[i]
      if par[i].find(":") != -1:
        var = par[i].split(":")[1]
      if (log[i].lower() == 'lin'):
        likel1 = likel1 + var
      if (log[i].lower() == 'log'):
        likel1 = likel1 + "l" + var
        likel2 = likel2 + var
        likel3 = likel3 + "10**(l" + var + ")"
        ilog = ilog + 1
        if (ilog < nlog):
          likel2 = likel2 + ","
          likel3 = likel3 + ","
      if (par[i][0] == 'f' and par[i] != 'fc'):
        if (len(species_r) == 1):
          likel4 = likel4 + "f0[0]"
        else:
          likel4 = likel4 + "f0[species_r=='" + par[i][1:].lower() + "']"
        likel5 = likel5 + par[i]
        igas = igas + 1
        if (igas < ngas):
          likel4 = likel4 + ","
          likel5 = likel5 + ","
    if (iret < nret-1):
      likel1 = likel1 + ","
    if (iret == nret-1):
      likel1 = likel1 + " = x"
    iret = iret + 1
targetl1 = likel1
targetl2 = "# not performing clr retrieval"
if clr:
  targetl2 = "xi   = np.float32([" + likel6 + "])\n  clrs = np.sum(np.exp(xi)) + np.exp(-np.sum(xi))"
targetl3 = "# no log-retrieved parameters"
if (nlog > 0):
  targetl3 = likel2 + " = " + likel3
targetl4 = "# no retrieved gases"
if (ngas > 0):
  targetl4 = likel4 + " = " + likel5

# set up prior syntax
targetp1 = targetl1
targetp2 = targetl3
if (clr and nlog-nlgas >= 1):
  targetp2 = priorl1 + " = " + priorl2
if (clr and nlog-nlgas  < 1):
  targetp2 = "# no log-retrieved non-gas parameters"
targetp3 = "# no retrieved gases or performing clr retrieval"
if (not clr):
  targetp3 = targetl4
iret    = 0
ilog    = 0
igas    = 0
priorl1 = "lng = 0.0 "
priorl2 = "if "
for i in range(npar):
  if (ret[i].lower() == 'y'):
    if (par[i][0] == 'f' and par[i] != 'fc' and clr):
      priorl2 = priorl2 + "ximin <= xi" + par[i][1:] + " <= ximax and "
    else:
      var = par[i]
      if par[i].find(":") != -1:
        var = par[i].split(":")[1]
      if (shp[i].lower() == 'g'):
        priorl1 = priorl1 + "- 0.5*(" + var + " - " + str(p1[i]) + ")**2/" + str(p2[i]) + "**2"
      if (shp[i].lower() == 'f'):
        priorl2 = priorl2 + str(p1[i]) + " <= " + var + " <= " + str(p2[i]) + " and "
if (not clr):
  priorl2 = priorl2 + "np.sum(f0) <= 1" + otherp
targetp4 = priorl1
targetp5 = priorl2

# read in base file
with open('t.rfast_retrieve', 'r') as file :
  filedata = file.read()

# replace targets with correct syntax
filedata = filedata.replace('TARGETG1',targetg1)
filedata = filedata.replace('TARGETG2',targetg2)
filedata = filedata.replace('TARGETG3',targetg3)
filedata = filedata.replace('TARGETL1',targetl1)
filedata = filedata.replace('TARGETL2',targetl2)
filedata = filedata.replace('TARGETL3',targetl3)
filedata = filedata.replace('TARGETL4',targetl4)
filedata = filedata.replace('TARGETP1',targetp1)
filedata = filedata.replace('TARGETP2',targetp2)
filedata = filedata.replace('TARGETP3',targetp3)
filedata = filedata.replace('TARGETP4',targetp4)
filedata = filedata.replace('TARGETP5',targetp5)
filedata = filedata.replace('TARGETUT1',targetut1)
filedata = filedata.replace('TARGETUA1',targetuA1)
filedata = filedata.replace('TARGETUO1',targetuo1)
filedata = filedata.replace('TARGETUC1',targetuc1)
filedata = filedata.replace('TARGETUT2',targetut2)
filedata = filedata.replace('TARGETUA2',targetuA2)
filedata = filedata.replace('TARGETUO2',targetuo2)
filedata = filedata.replace('TARGETUC2',targetuc2)

# write new file
with open('rfast_retrieve.py', 'w') as file:
  file.write(filedata)

# similar to above, except for analysis code
targeta1 = targetg1
if clr:
  targeta1 = targetg1.replace("xi","lf").replace("/gx","").replace("log","log10")
targeta2 = targetg2
targeta4 = targetg3.replace("guess = [","truths = [")
if clr:
  targeta4 = targeta4.replace("xi","lf")
targeta5 = "gind = ["
iret   = 0
igas   = 0
anlyz1 = "names  = ["
for i in range(npar):
  if (ret[i].lower() == 'y'):
    if (log[i].lower() == 'log'):
      anlyz1 = anlyz1 + 'r"$\log\,$"+' + lab[i]
    else:
      anlyz1 = anlyz1 + lab[i]
    if (iret < nret-1):
      anlyz1 = anlyz1 + ","
    if (iret == nret-1):
      anlyz1 = anlyz1 + "]"
    if (par[i][0] == 'f' and par[i] != 'fc' and clr):
      targeta5 = targeta5 + str(iret)
      igas = igas + 1
      if (igas < ngas):
        targeta5 = targeta5 + ","
    iret = iret + 1
targeta3 = anlyz1
targeta5 = targeta5 + "]"
targeta6 = targetl1.replace(" = x"," = samples[pos_max][0]")
if clr:
  targeta6 = targeta6.replace("xi","lf")
targeta7 = targetl3
if clr:
  targeta7 = targeta7.replace("np.exp(xi","10**(lf").replace("/clrs","")
targeta8 = targetl4

# read in base file
with open('t.rfast_analyze', 'r') as file :
  filedata = file.read()

# replace targets with correct syntax
filedata = filedata.replace('TARGETA1',targeta1)
filedata = filedata.replace('TARGETA2',targeta2)
filedata = filedata.replace('TARGETA3',targeta3)
filedata = filedata.replace('TARGETA4',targeta4)
filedata = filedata.replace('TARGETA5',targeta5)
filedata = filedata.replace('TARGETA6',targeta6)
filedata = filedata.replace('TARGETA7',targeta7)
filedata = filedata.replace('TARGETA8',targeta8)
filedata = filedata.replace('TARGETUT1',targetut1)
filedata = filedata.replace('TARGETUA1',targetuA1)
filedata = filedata.replace('TARGETUO1',targetuo1)
filedata = filedata.replace('TARGETUC1',targetuc1)
filedata = filedata.replace('TARGETUT2',targetut2)
filedata = filedata.replace('TARGETUA2',targetuA2)
filedata = filedata.replace('TARGETUO2',targetuo2)
filedata = filedata.replace('TARGETUC2',targetuc2)

# write new file
with open('rfast_analyze.py', 'w') as file:
  file.write(filedata)

# document retrieved parameters to file
shutil.copy(filename_ret,dirout+fnr+'_rpars.log')
if not os.path.isfile(dirout+filename_ret):
  shutil.copy(filename_ret,dirout+filename_ret)