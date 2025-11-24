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
    
    
def wsclean_cmd(minuv, size, briggs, taper, datacol, name, scale, niter, ms, outname, mask = ''):
    cmd = f'wsclean -no-update-model-required -minuv-l {minuv} -size {size} {size} \
            -reorder -weight briggs {briggs} \
            -clean-border 1 -mgain 0.8 -fit-beam -data-column {datacol} \
            -join-channels -channels-out 6 -padding 1.4 -multiscale \
            -multiscale-scales 0,4,8,16,32 -fit-spectral-pol 3 -pol i \
            -baseline-averaging 8.52211548825 -name {outname} \
            -scale {scale}arcsec -niter {niter} '
    if mask != '':
        if name != '':
            cmd += f'-fits-mask {name}_{mask}-MFS-image.mask.fits -auto-threshold 2.5 '
        else:
            cmd += f'-fits-mask {mask}-MFS-image.mask.fits -auto-threshold 2.5 '
    if taper != 0:
        cmd += f'-taper-gaussian {taper}arcsec '
    cmd += f'{ms} >log.txt'
    return cmd


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
dir_sub_shallow = os.path.join(dir_img, 'sub_shallow')
dir_sub_deep = os.path.join(dir_img, 'sub_deep')
dir_uvcut_shallow = os.path.join(dir_img, 'uvcut_shallow')
dir_uvcut_deep = os.path.join(dir_img, 'uvcut_deep')
dir_halo_hr_shallow = os.path.join(dir_img, 'halo_hr_shallow')
dir_halo_hr_deep = os.path.join(dir_img, 'halo_hr_deep')
dir_halo_lr_shallow = os.path.join(dir_img, 'halo_lr_shallow')
dir_halo_lr_deep = os.path.join(dir_img, 'halo_lr_deep')

dirs_to_create = [dir_shallow_img, dir_deep_img, dir_exp_shallow, dir_exp_deep,
                dir_sub_shallow, dir_sub_deep, dir_uvcut_shallow, dir_uvcut_deep,
                dir_halo_hr_shallow, dir_halo_hr_deep, dir_halo_lr_shallow, dir_halo_lr_deep]
for d in dirs_to_create:
    directory(d)

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
only_sub = int(variables['only_sub'])
minuv_sub = int(variables['minuv_sub'])

ms = os.path.join(dir_mss, mss_name)
#### Imaging
### only discrete sources image


ts  = pt.table(f'{ms}', readonly=False)
colnames = ts.colnames()
ts.close()   
if 'exponential' in colnames:
    logger.info(f"Column 'exponential' already exists in MS: {ms}")
else:
    logger.info(f"Adding column 'exponential' to MS: {ms}")
    cmd = f'DP3 msin={ms} msout=. steps=[] msout.datacolumn=exponential \
            msin.datacolumn=DATA msout.storagemanager=dysco >log_add_column.txt'
    logger.info(f"Command to add column: {cmd}")
    os.system(cmd)
    logger.info(f"Column 'exponential' added to MS: {ms}") 

logger.info(f"Injecting exponential in a single column")
predict_cmd = f'wsclean -predict -name {dir_img}/exponential {dir_mss}/{mss_name} \
                >log_predict_exponential.txt'
logger.info(f"Command to predict visibilities for exponential: {predict_cmd}")
os.system(predict_cmd)
logger.info("Exponential model predicted.")
logger.info("Injecting exponential into the column 'exponential'...")

ts = pt.table(ms, readonly=False)
colnames = ts.colnames()
stepsize = 10000
data_column = 'DATA'
for row in range(0, ts.nrows(), stepsize):
    print(f"Doing {row} out of {ts.nrows()}, (step: {stepsize})")
    data  = ts.getcol(data_column, startrow=row, nrow=stepsize, rowincr=1)
    model = ts.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
    ts.putcol('exponential', data+model, startrow=row, nrow=stepsize, rowincr=1)
ts.close()
logger.info("Model prediction and injection completed successfully.")

logger.info('Imaging the radio halo at low resolution...')
os.chdir(dir_halo_lr_shallow)
logger.info('Source subtracted shallow image...')

shallow_cmd = wsclean_cmd(minuv = 80, size = 480, briggs = -0.5, taper = 60,
                        datacol = 'exponential', name = 'halo_lr', scale = 6, niter = 15000, 
                        ms = ms, outname = 'halo_lr_shallow')
logger.info(f"Running shallow imaging command: {shallow_cmd}")
os.system(shallow_cmd)
logger.info('Shallow image created.')
    
breizorro_shallow = f'breizorro.py -t 3 -r halo_lr_shallow-MFS-image.fits'
logger.info(f"Making mask: {breizorro_shallow}")
os.system(breizorro_shallow)
move_mask = f'mv *.mask.fits {dir_halo_lr_deep}/'
logger.info(f"Moving mask to deep image directory: {move_mask}")
os.system(move_mask)

os.chdir(dir_halo_lr_deep)
deep_cmd = wsclean_cmd(minuv = 80, size = 480, briggs = -0.5, taper = 60,
                        datacol = 'exponential', name = 'halo_lr', scale = 6, niter = 10000000,
                        outname = 'halo_lr_deep', ms = ms, mask = 'shallow')
logger.info(f"Running deep imaging command: {deep_cmd}")
os.system(deep_cmd)
logger.info('Deep image created.')

logger.info("All imaging and subtraction tasks completed successfully.")

if not only_sub:
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
    
    
    os.chdir(dir_shallow_img)
    shallow_cmd = wsclean_cmd(minuv = 80, size = 480, briggs = -0.5, taper = 60, 
                            datacol = 'inj', name = name, scale = 6, niter = 15000, 
                            ms = ms, outname = name + '_synth_shallow')
    logger.info(f"Running shallow imaging command: {shallow_cmd}")
    os.system(shallow_cmd)

    breizorro_shallow = f'breizorro.py -t 3 -r {name}_synth_shallow-MFS-image.fits'
    logger.info(f"Making mask: {breizorro_shallow}")
    os.system(breizorro_shallow)

    move_mask = f'mv *.mask.fits {dir_deep_img}/'
    logger.info(f"Moving mask to deep image directory: {move_mask}")
    os.system(move_mask)

    os.chdir(dir_deep_img)
    deep_cmd = wsclean_cmd(minuv = 80, size = 480, briggs = -0.5, taper = 60, 
                        datacol = 'inj', name = name, scale = 6, niter = 10000000,
                        outname = name + 'synth_deep', ms = ms, mask = 'synth_shallow')
    logger.info(f"Running deep imaging command: {deep_cmd}")
    os.system(deep_cmd)

    # Imaging with injected exponential source
    os.chdir(dir_exp_shallow)
    exp_shallow = wsclean_cmd(minuv = 80, size = 480, briggs = -0.5, taper = 60, 
                            datacol = 'inj_exp', name = name, scale = 6, niter = 15000, 
                            ms = ms, outname = name + '_exp_shallow')
    logger.info(f"Running shallow imaging command: {exp_shallow}")
    os.system(exp_shallow)

    breizorro_shallow = f'breizorro.py -t 3 -r {name}_exp_shallow-MFS-image.fits'
    logger.info(f"Making mask: {breizorro_shallow}")
    os.system(breizorro_shallow)

    move_mask = f'mv *.mask.fits {dir_exp_deep}/'
    logger.info(f"Moving mask to deep image directory: {move_mask}")
    os.system(move_mask)

    os.chdir(dir_exp_deep)
    exp_deep = wsclean_cmd(minuv = 80, size = 480, briggs = -0.5, taper = 60, 
                           datacol = 'inj_exp', name = name, scale = 6, niter = 10000000,
                           ms = ms, outname = name + '_exp_deep', mask = 'exp_shallow')
    logger.info(f"Running deep imaging command: {exp_deep}")
    os.system(exp_deep)
    
else:
    logger.info('Skipping injection and imaging of sources, only subtraction and following steps will be performed.')

os.chdir(dir_uvcut_shallow)    
img_sub_shallow = wsclean_cmd(minuv = minuv_sub, size = 1920, briggs = -0.5, taper = 0, 
                            datacol = 'inj_exp', name = '', scale = 1.5, niter = 15000, 
                            ms = ms, outname = 'highcut_shallow')
logger.info(f"Running shallow image for subtraction command: {img_sub_shallow}")
os.system(img_sub_shallow)
logger.info('Shallow image created.')

breizorro_shallow = f'breizorro.py -t 5 -r highcut_shallow-MFS-image.fits'
logger.info(f"Making mask for shallow image: {breizorro_shallow}")
os.system(breizorro_shallow)
move_mask = f'mv *.mask.fits {dir_uvcut_deep}/'
logger.info(f"Moving mask to deep image directory: {move_mask}")
os.system(move_mask)

os.chdir(dir_uvcut_deep)
img_sub_deep = wsclean_cmd(minuv = minuv_sub, size = 1920, briggs = -0.5, taper = 0, 
                           datacol = 'inj_exp', name = '', scale = 1.5, niter = 10000000,
                           ms = ms, outname = 'highcut_deep', mask = 'highcut_shallow')
logger.info(f"Running deep image for subtraction command: {img_sub_deep}")
os.system(img_sub_deep)
logger.info('Deep image created.')

logger.info(f"Predicting visibilities for model 'high_cut_deep' in MS: {mss_name}")
predict_cmd = f'wsclean -predict -name {dir_uvcut_deep}/highcut_deep -channels-out 6 {dir_mss}/{mss_name} \
                >log_predict.txt'
logger.info(f"Command to predict visibilities: {predict_cmd}")
os.system(predict_cmd)
logger.info("Model predicted.")

ts = pt.table(ms, readonly=False)
colnames = ts.colnames()
logger.info("Starting model subtraction from MS...")
stepsize = 10000
data_column = 'inj_exp'
outcolumn = 'sub'
for row in range(0, ts.nrows(), stepsize):
    print(f"Doing {row} out of {ts.nrows()}, (step: {stepsize})")
    data  = ts.getcol(data_column, startrow=row, nrow=stepsize, rowincr=1)
    model = ts.getcol('MODEL_DATA', startrow=row, nrow=stepsize, rowincr=1)
    ts.putcol(outcolumn, data-model, startrow=row, nrow=stepsize, rowincr=1)
ts.close()
logger.info("Subtraction completed successfully.")


os.chdir(dir_sub_shallow)
logger.info('Source subtracted shallow image...')

shallow_cmd = wsclean_cmd(minuv = 80, size = 480, briggs = -0.5, taper = 60,
                        datacol = 'sub', name = 'A2244', scale = 6, niter = 15000, 
                        ms = ms, outname = name + '_sub_shallow')
logger.info(f"Running shallow imaging command: {shallow_cmd}")
os.system(shallow_cmd)
logger.info('Shallow image created.')

breizorro_shallow = f'breizorro.py -t 3 -r {name}_sub_shallow-MFS-image.fits'
logger.info(f"Making mask: {breizorro_shallow}")
os.system(breizorro_shallow)
move_mask = f'mv *.mask.fits {dir_sub_deep}/'
logger.info(f"Moving mask to deep image directory: {move_mask}")
os.system(move_mask)

os.chdir(dir_sub_deep)
deep_cmd = wsclean_cmd(minuv = 80, size = 480, briggs = -0.5, taper = 60,
                        datacol = 'sub', name = 'A2244', scale = 6, niter = 10000000,
                        outname = name + '_sub_deep', ms = ms, mask = 'sub_shallow')
logger.info(f"Running deep imaging command: {deep_cmd}")
os.system(deep_cmd)
logger.info('Deep image created.')


os.chdir(dir_halo_hr_shallow)
logger.info('Source subtracted shallow image...')

shallow_cmd = wsclean_cmd(minuv = 80, size = 1920, briggs = -0.5, taper = 0,
                        datacol = 'inj_exp', name = 'halo_hr', scale = 1.5, niter = 15000, 
                        ms = ms, outname = 'halo_hr_shallow')
logger.info(f"Running shallow imaging command: {shallow_cmd}")
os.system(shallow_cmd)
logger.info('Shallow image created.')

breizorro_shallow = f'breizorro.py -t 3 -r halo_hr_shallow-MFS-image.fits'
logger.info(f"Making mask: {breizorro_shallow}")
os.system(breizorro_shallow)
move_mask = f'mv *.mask.fits {dir_halo_hr_deep}/'
logger.info(f"Moving mask to deep image directory: {move_mask}")
os.system(move_mask)

os.chdir(dir_halo_hr_deep)
deep_cmd = wsclean_cmd(minuv = 80, size = 1920, briggs = -0.5, taper = 0,
                        datacol = 'inj_exp', name = 'halo_hr', scale = 1.5, niter = 10000000,
                        outname = 'halo_hr_deep', ms = ms, mask = 'shallow')
logger.info(f"Running deep imaging command: {deep_cmd}")
os.system(deep_cmd)
logger.info('Deep image created.')

logger.info("All imaging and subtraction tasks completed successfully.")