#!/usr/bin/env python

import os
import logging
import casacore.tables as pt

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

logger.info("Starting model prediction and injection script...")

dir_work = os.getcwd()
dir_mss = os.path.join(dir_work, 'mss')
dir_img = os.path.join(dir_work, 'img')
dir_parsets = os.path.join(dir_work, 'parsets')
inj_parset = os.path.join(dir_parsets, 'inj.parset')

try:
    with open(inj_parset, 'r') as file:
        variables = {}
        for line in file:
            line = line.strip()                         # Removes spaces and newlines
            if '=' in line:
                key, value = line.split('=', 1)         # Splits on the first '='
                variables[key.strip()] = value.strip()  # adds to the dictionary
except FileNotFoundError:
    logger.error(f"Parset file not found at {inj_parset}")
    # Exit if parset file is missing

model_name = variables['model_name']
mss_name = variables['mssname']
outcolumn = variables['outcolumn']
data_column = variables['data_column']

logger.info(f"Predicting visibilities for model: {model_name} in MS: {mss_name}")
predict_cmd = f'wsclean -predict -name {dir_img}/{model_name} {dir_mss}/{mss_name} \
                >log_predict.txt'
logger.info(f"Command to predict visibilities: {predict_cmd}")
os.system(predict_cmd)
logger.info("Visibilities prediction command executed.")


mslist = [os.path.join(dir_mss, mss_name)]
logger.info(f"Adding column '{outcolumn}' to MS: {mss_name}")
for i in range(len(mslist)):
    cmd = f'DP3 msin={mslist[i]} + msout=. steps=[] msout.datacolumn={outcolumn} \
            msin.datacolumn=DATA msout.storagemanager=dysco >log_add_column.txt'
    logger.info(f"Command to add column: {cmd}")
    os.system(cmd)
logger.info(f"Column '{outcolumn}' added to MS: {mss_name}")

logger.info("Starting model injection into MS...")
stepsize = 10000
for ms in mslist:
    ts  = pt.table(ms, readonly=False)
    colnames = ts.colnames()
    if 'CORRECTED_DATA' in colnames:
        for row in range(0, ts.nrows(), stepsize):
            print(f"Doing {row} out of {ts.nrows()}, (step: {stepsize})")
            print('Read CORRECTED_DATA column')
            data  = ts.getcol('CORRECTED_DATA', startrow=row, nrow=stepsize, rowincr=1)
            print('Reading MODEL column')
            model = ts.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
            print('Subtraction...')
            ts.putcol(outcolumn, data+model, startrow=row, nrow=stepsize, rowincr=1)
    else:
        for row in range(0, ts.nrows(), stepsize):
            print(f"Doing {row} out of {ts.nrows()}, (step: {stepsize})")
            data  = ts.getcol(data_column, startrow=row, nrow=stepsize, rowincr=1)
            model = ts.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
            ts.putcol(outcolumn, data+model, startrow=row, nrow=stepsize, rowincr=1)
    ts.close()
logger.info("Model prediction and injection completed successfully.")