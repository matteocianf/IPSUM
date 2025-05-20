#!/usr/bin/env python

import os
import math

dir_work = os.getcwd()
dir_mss = os.path.join(dir_work, 'mss')
synthms_parset = os.path.join(dir_work, 'synthms.parset')

with open(synthms_parset, 'r') as file:
    variables = {}
    for line in file:
        line = line.strip()  # Removes spaces and newlines
        if '=' in line:
            key, value = line.split('=', 1)  # Splits on the first '='
            variables[key.strip()] = value.strip()  # adds to the dictionary
            
# Parameters
name = variables['name']
tobs = float(variables['tobs'])                  # observation time in hours
station = variables['station']                   # HBA, LBA or both
minfreq = float(variables['minfreq'])            # minimum frequency in MHz
maxfreq = float(variables['maxfreq'])            # maximum frequency in MHz
lofarversion = int(variables['lofarversion'])    # LOFAR version, 1 or 2
ra = float(variables['ra'])                      # RA in degrees
dec = float(variables['dec'])                    # DEC in degrees

ra = math.radians(ra)
dec = math.radians(dec)

# os.chdir(dir_mss)
cmd = f'./synthms --name {name} --tobs {tobs} --station {station} --minfreq {minfreq*1e6} --maxfreq {maxfreq*1e6} --lofarversion {lofarversion} --ra {ra} --dec {dec}'
print(cmd)
os.system(cmd)

os.chdir(dir_work)