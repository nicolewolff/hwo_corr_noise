import os
import numpy as np

def calc_Correlated_SNR(snr_tot, snr_uncorr):
    snr_corr = ((1/snr_tot**2) - (1/snr_uncorr**2))**(-1/2)
    return snr_corr

def calc_GP_Amplitude(snr_corr):
    amp = 0.23899871939675468 * 10 / snr_corr  ### .279 and 10 are a pre-calculated amplitude/SNR pair for SNR calc at 0.55um
    return amp

### parameters ### 
snr = np.sqrt(800)
snr_tot = np.sqrt(200)
length = '4e-3'

snr_corr = calc_Correlated_SNR(snr_tot, snr)
gp_amp = calc_GP_Amplitude(snr_corr)

### running rfast ###
for iteration in range (1,5):
    fr = open('rfast_inputs.scr', 'r+')
    lines = fr.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith('snr0'):
            lines[i] = "snr0  = "+str(snr)+"  \n"
        
        elif line.startswith('dirout'):
            lines[i] = "dirout  = outputs_correlated_052325/high_res_snr_uncorr_{:.2f}".format(snr)+"_snr_tot_{:.2f}".format(snr_tot)+"_length_"+length+"/"+str(iteration)+"   \n"
        
        elif line.startswith('arho'):
            lines[i] = "arho    = " + str(gp_amp)+ "    \n"   
        
    fr.close()

    with open('rfast_inputs.scr', 'w') as file:
        file.writelines(lines)
    
    os.system("sh run.sh")

    