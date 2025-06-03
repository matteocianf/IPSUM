#!/usr/bin/env python

import os
import logging
import casacore.tables as pt

# Set up logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('pred_inj.log', 'w+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def directory(directory):
   if not os.path.exists(directory):
    os.mkdir(directory)
    logger.info(f"Directory {directory} created")
   else:
    logger.info(f"Directory {directory} already exists")


logger.info("Starting model prediction and injection script...")

dir_work = os.getcwd()
dir_mss = os.path.join(dir_work, 'mss')
dir_img = os.path.join(dir_work, 'img')
dir_parsets = os.path.join(dir_work, 'parsets')
inj_parset = os.path.join(dir_parsets, 'inj.parset')

dir_shallow_img = os.path.join(dir_img, 'synth_shallow')
dir_deep_img = os.path.join(dir_img, 'synth_deep')
dir_exp_shallow = os.path.join(dir_img, 'synth_exp_shallow')
dir_exp_deep = os.path.join(dir_img, 'synth_exp_deep')
directory(dir_shallow_img)
directory(dir_deep_img)
directory(dir_exp_shallow)
directory(dir_exp_deep)

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

mss_name = variables['mssname']
name = variables['name']

ms = os.path.join(dir_mss, mss_name)
cols = ['inj', 'inj_exp']
models = ['inj_sources', 'exponential']
for col in cols:
    model_name = models[0] if col == 'inj' else models[1]
    logger.info(f"Predicting visibilities for model: {model_name} in MS: {mss_name}")
    predict_cmd = f'wsclean -predict -name {dir_img}/{model_name} {dir_mss}/{mss_name} \
                    >log_predict.txt'
    logger.info(f"Command to predict visibilities: {predict_cmd}")
    os.system(predict_cmd)
    logger.info("Model predicted.")

    ts = pt.table(ms, readonly=False)
    colnames = ts.colnames()
    logger.info("Starting model injection into MS...")
    stepsize = 10000
    data_column = 'DATA' if col == 'inj' else 'inj'
    for row in range(0, ts.nrows(), stepsize):
        print(f"Doing {row} out of {ts.nrows()}, (step: {stepsize})")
        data  = ts.getcol(data_column, startrow=row, nrow=stepsize, rowincr=1)
        model = ts.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
        ts.putcol(col, data+model, startrow=row, nrow=stepsize, rowincr=1)
    ts.close()
logger.info("Model prediction and injection completed successfully.")


#### Imaging
### only discrete sources image
os.chdir(dir_shallow_img)
shallow_cmd = f'wsclean -no-update-model-required -minuv-l 80 -size 480 480 \
                -reorder -weight briggs -0.5 -taper-gaussian 60arcsec \
            	-clean-border 1 -mgain 0.8 -fit-beam -data-column inj \
	            -join-channels -channels-out 6 -padding 1.4 -multiscale \
	            -multiscale-scales 0,4,8,16,32,64 -fit-spectral-pol 3 -pol i \
	            -baseline-averaging 8.52211548825 -name {name}_synth_shallow \
	            -scale 6arcsec -niter 15000 \
	            {ms} \
	            >log.txt'
logger.info(f"Running shallow imaging command: {shallow_cmd}")
# os.system(shallow_cmd)

breizorro_shallow = f'breizorro.py -t 3 -r {name}_synth_shallow-MFS-image.fits'
logger.info(f"Making mask: {breizorro_shallow}")
# os.system(breizorro_shallow)

move_mask = f'mv *.mask.fits {dir_deep_img}/'
logger.info(f"Moving mask to deep image directory: {move_mask}")
# os.system(move_mask)

os.chdir(dir_deep_img)
deep_cmd = f'wsclean -no-update-model-required -minuv-l 80 -size 480 480 \
            -reorder -weight briggs -0.5 -taper-gaussian 60arcsec \
            -clean-border 1 -mgain 0.8 -fit-beam -data-column inj \
            -join-channels -channels-out 6 -padding 1.4 -multiscale \
            -multiscale-scales 0,4,8,16,32,64 -fit-spectral-pol 3 -pol i \
            -baseline-averaging 8.52211548825 -name {name}_synth \
            -scale 6arcsec -niter 100000 -fits-mask {name}_synth_shallow-MFS-image.mask.fits\
            {ms} \
            >log.txt'
logger.info(f"Running deep imaging command: {deep_cmd}")
# os.system(deep_cmd)

os.chdir(dir_exp_shallow)
exp_shallow = f'wsclean -no-update-model-required -minuv-l 80 -size 480 480 \
                -reorder -weight briggs -0.5 -taper-gaussian 60arcsec \
                -clean-border 1 -mgain 0.8 -fit-beam -data-column inj_exp \
                -join-channels -channels-out 6 -padding 1.4 -multiscale \
                -multiscale-scales 0,4,8,16,32,64 -fit-spectral-pol 3 -pol i \
                -baseline-averaging 8.52211548825 -name {name}_exp_shallow \
                -scale 6arcsec -niter 15000 \
                {ms} \
                >log.txt'
logger.info(f"Running shallow imaging command: {exp_shallow}")
# os.system(exp_shallow)

breizorro_shallow = f'breizorro.py -t 3 -r {name}_exp_shallow-MFS-image.fits'
logger.info(f"Making mask: {breizorro_shallow}")
# os.system(breizorro_shallow)

move_mask = f'mv *.mask.fits {dir_exp_deep}/'
logger.info(f"Moving mask to deep image directory: {move_mask}")
# os.system(move_mask)

os.chdir(dir_exp_deep)
exp_deep = f'wsclean -no-update-model-required -minuv-l 80 -size 480 480 \
            -reorder -weight briggs -0.5 -taper-gaussian 60arcsec \
            -clean-border 1 -mgain 0.8 -fit-beam -data-column inj_exp \
            -join-channels -channels-out 6 -padding 1.4 -multiscale \
            -multiscale-scales 0,4,8,16,32,64 -fit-spectral-pol 3 -pol i \
            -baseline-averaging 8.52211548825 -name {name}_exp \
            -scale 6arcsec -niter 100000 -fits-mask {name}_exp_shallow-MFS-image.mask.fits \
            {ms} \
            >log.txt'
logger.info(f"Running deep imaging command: {exp_deep}")
# os.system(exp_deep)

# to do img shallow mask img deep img exp shallow mask img exp deep (predict?) subtraction disc sources sub shallow img mask sub deep img 