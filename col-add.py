#!/usr/bin/env python

import os

dir = "/local/work/"
mslist = [dir + "synth.ms"]
outcolumn = "inj"
for i in range(len(mslist)):
    cmd = 'DP3 msin=' + mslist[i] + \
        ' msout=. steps=[] msout.datacolumn=' + outcolumn + ' '
    cmd += 'msin.datacolumn=DATA msout.storagemanager=dysco'
    os.system(cmd)
