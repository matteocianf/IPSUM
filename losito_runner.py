#!/usr/bin/env python

########################################
# Author: Matteo Cianfaglione          #
# e-mail: matteo.cianfaglione@unibo.it #
########################################


import os
import re
import math
import logging
import casacore as pt

# Set up logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('source_generator.log', 'w+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Starting losito_runner script...")

dir_work = os.getcwd()
dir_mss = os.path.join(dir_work, 'mss')
dir_mss_bkp = os.path.join(dir_work, 'mss-bkp')
dir_parsets = os.path.join(dir_work, 'parsets')
synthms_parset = os.path.join(dir_parsets, 'synthms.parset')

logger.info(f"Working directory: {dir_work}")
logger.info(f"MS directory: {dir_mss}")
logger.info(f"Backup MS directory: {dir_mss_bkp}")
logger.info(f"Parsets directory: {dir_parsets}")

if not os.path.exists(dir_mss):
    os.mkdir(dir_mss)
    logger.info(f"Directory {dir_mss} created")
else:
    logger.info(f"Directory {dir_mss} already exists")

try:
    with open(synthms_parset, 'r') as file:
        variables = {}
        for line in file:
            line = line.strip()                         # Removes spaces and newlines
            if '=' in line:
                key, value = line.split('=', 1)         # Splits on the first '='
                variables[key.strip()] = value.strip()  # adds to the dictionary
except FileNotFoundError:
    logger.error(f"Parset file not found at {synthms_parset}")
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
remove_SB = bool(variables['remove_last_SB'])    # remove last subband
only_add_col = bool(variables['only_add_col'])   # only add columns to MS

ra = math.radians(ra)
dec = math.radians(dec)

if only_add_col == True:
    ms = os.path.join(dir_mss, name)
    cols = ['inj', 'inj_exp']
    ts  = pt.table(ms, readonly=False)
    colnames = ts.colnames()
    for col in cols:
        logger.info(f"Adding column '{col}' to MS: {name}")
        if col in colnames:
            logger.info(f"Column '{col}' already exists in MS: {name}")
            continue
        else:
            logger.info(f"Adding column '{col}' to MS: {name}")
            cmd = f'DP3 msin={ms} msout=. steps=[] msout.datacolumn={col} \
                    msin.datacolumn=DATA msout.storagemanager=dysco >log_add_column.txt'
            logger.info(f"Command to add column: {cmd}")
            os.system(cmd)
            logger.info(f"Column '{col}' added to MS: {name}") 
    ts.close()   
    
else:
    os.chdir(dir_mss)
    cmd = f'synthms --name {name} --tobs {tobs} --station {station} --minfreq {minfreq*1e6} --maxfreq {maxfreq*1e6} \
            --lofarversion {lofarversion} --ra {ra} --dec {dec} --chanpersb {chanpersb} --tres {tres} --start {start}'
    logger.info(f"Running command: {cmd}")
    os.system(cmd)

    if not os.path.exists(dir_mss):
        copy_cmd = f'cp -r {dir_mss} {dir_mss_bkp}'
        os.system(copy_cmd)
        logger.info(f"Backup directory {dir_mss_bkp} created")
    else:
        logger.info(f"Backup directory {dir_mss_bkp} already exists")

    losito_run = f'losito {dir_parsets}/losito.parset >log_losito.txt'
    logger.info(f"Running losito with command: {losito_run}")
    os.system(losito_run)

    if remove_SB:
        num = []
        for file in os.listdir(dir_mss):
            if file.endswith('.MS'):
                match = re.search(r'\d+', file)
                if match:
                    num.append(int(match.group()))
                else:
                    logger.warning(f"No number found in file name: {file}")
        last_sb = max(num) if num else None
        remove_last_SB = f'rm -r *{last_sb}.MS'
        logger.info(f"Removing last subband with command: {remove_last_SB}")
        os.system(remove_last_SB)

    single_ms = f'DP3 msin={name}*.MS msout={name}.MS msout.storagemanager=dysco steps=[] >log_dp3_unifier.txt'
    logger.info(f"Running DP3 with command: {single_ms}")
    os.system(single_ms)

    single_ms_bkp = f'cp -r {name}.MS {dir_mss_bkp}/{name}.MS'
    logger.info(f"Backing up single MS with command: {single_ms_bkp}")
    os.system(single_ms_bkp)

    freq_avg = f'DP3 {dir_parsets}/dp3_freqavg.parset >log_dp3_freqavg.txt'
    logger.info(f"Running DP3 frequency averaging with command: {freq_avg}")
    os.system(freq_avg)

    ms = os.path.join(dir_mss, name)
    cols = ['inj', 'inj_exp']
    ts  = pt.table(ms, readonly=False)
    colnames = ts.colnames()
    for col in cols:
        logger.info(f"Adding column '{col}' to MS: {name}")
        if col in colnames:
            logger.info(f"Column '{col}' already exists in MS: {name}")
            continue
        else:
            logger.info(f"Adding column '{col}' to MS: {name}")
            cmd = f'DP3 msin={ms} msout=. steps=[] msout.datacolumn={col} \
                    msin.datacolumn=DATA msout.storagemanager=dysco >log_add_column.txt'
            logger.info(f"Command to add column: {cmd}")
            os.system(cmd)
            logger.info(f"Column '{col}' added to MS: {name}") 
    ts.close()   
    
os.chdir(dir_work)
logger.info("losito_runner script completed successfully.")