#!/usr/bin/env python

import os
import casacore.tables as pt

dir_work = os.getcwd()
dir_mss = os.path.join(dir_work, 'mss')
dir_img = os.path.join(dir_work, 'img')
inj_parset = os.path.join(dir_work, 'inj.parset')

try:
    with open(inj_parset, 'r') as file:
        variables = {}
        for line in file:
            line = line.strip()                         # Removes spaces and newlines
            if '=' in line:
                key, value = line.split('=', 1)         # Splits on the first '='
                variables[key.strip()] = value.strip()  # adds to the dictionary
except FileNotFoundError:
    print(f"Error: Parset file not found at {inj_parset}")
    # Exit if parset file is missing

model_name = variables['model_name']
mss_name = variables['mss_name']

predict_cmd = 'wsclean -predict -name {model_name} {dir_mss}/{mss_name} \
                > log_predict.txt'

mslist = [dir_mss + mss_name]
outcolumn = "inj"
for i in range(len(mslist)):
    cmd = f'DP3 msin={mslist[i]} + msout=. steps=[] msout.datacolumn={outcolumn} \
            msin.datacolumn=DATA msout.storagemanager=dysco'
    os.system(cmd)


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
            ts.putcol(outcolumn, data-model, startrow=row, nrow=stepsize, rowincr=1)
    else:
        for row in range(0, ts.nrows(), stepsize):
            print(f"Doing {row} out of {ts.nrows()}, (step: {stepsize})")
            data  = ts.getcol('sub2', startrow=row, nrow=stepsize, rowincr=1)
            model = ts.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
            ts.putcol(outcolumn, data+model, startrow=row, nrow=stepsize, rowincr=1)
    ts.close()
