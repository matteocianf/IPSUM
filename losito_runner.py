#!/usr/bin/env python

########################################
# Author: Matteo Cianfaglione          #
# e-mail: matteo.cianfaglione@unibo.it #
########################################


import os
import math

dir_work = os.getcwd()
dir_mss = os.path.join(dir_work, 'mss')
synthms_parset = os.path.join(dir_work, 'synthms.parset')

if not os.path.exists(dir_mss):
    os.mkdir(dir_mss)
    print(f"Directory {dir_mss} non existent, created")
else:
    print(f"Directory {dir_mss} found")

try:
    with open(synthms_parset, 'r') as file:
        variables = {}
        for line in file:
            line = line.strip()                         # Removes spaces and newlines
            if '=' in line:
                key, value = line.split('=', 1)         # Splits on the first '='
                variables[key.strip()] = value.strip()  # adds to the dictionary
except FileNotFoundError:
    print(f"Error: Parset file not found at {synthms_parset}")
    # Exit if parset file is missing
            
# Parameters
name = variables['name']
tobs = float(variables['tobs'])                  # observation time in hours
station = variables['station']                   # HBA, LBA or both
minfreq = float(variables['minfreq'])            # minimum frequency in MHz
maxfreq = float(variables['maxfreq'])            # maximum frequency in MHz
lofarversion = int(variables['lofarversion'])    # LOFAR version, 1 or 2
chanpersb = int(variables['chanpersb'])          # channels per subband
tres = float(variables['tres'])                  # time resolution in seconds
start = float(variables['start']) * 3600 * 24    # start time in MJD
ra = float(variables['ra'])                      # RA in degrees
dec = float(variables['dec'])                    # DEC in degrees

ra = math.radians(ra)
dec = math.radians(dec)

os.chdir(dir_mss)
cmd = f'synthms --name {name} --tobs {tobs} --station {station} --minfreq {minfreq*1e6} --maxfreq {maxfreq*1e6} --lofarversion {lofarversion} --ra {ra} --dec {dec} --chanpersb {chanpersb} --tres {tres} --start {start}'
print(cmd)
os.system(cmd)

losito_run = f'losito ../losito.parset'
print(losito_run)
os.system(losito_run)

freq_avg = f'DP3 ../dp3_freqavg.parset'
print(freq_avg)
os.system(freq_avg)

single_ms = f'DP3 msin={name}*.MS msout={name}.MS msout.storemanager=dysco'
print(single_ms)
os.system(single_ms)

os.chdir(dir_work)