# INTACT
INTACT (INjector of mock discreTe sources in gAlaxy ClusTers) -Pipeline to create a mock observation starting from a real one and injecting discrete sources in it. This is specifically thought for galaxy clusters

The current requirements are:
```
argparse
numpy
astropy
matplotlib
losito
```

This pipeline works by generating an empty LOFAR image with only noise in it using LoSiTo and then injecting in its MODEL_DATA column a uniform 3d distribution of sources (then projected into the 2d plane).

For now the various parameters are given via argparse, it is possible that I will change this in the future with the implementation of a parset file.
