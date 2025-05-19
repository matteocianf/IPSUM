# INTACT
INTACT (INjector of mock discreTe sources in gAlaxy ClusTers) -Pipeline to create a mock observation starting from a real one and injecting discrete sources in it. This is specifically thought for galaxy clusters

The current requirements are:
```
numpy
astropy
matplotlib
losito (and dependences therein)
```
This pipeline works by generating an empty LOFAR image with only noise in it using LoSiTo and then injecting in its MODEL_DATA column a uniform 3d distribution of sources (then projected into the 2d plane). Then the image can be produced with WSClean or other softwares for radio imaging.
The parameters are inputted via a file called `intact.parset`, there is an example file in the repo. It is important to notice that the imsize for the model must be the same as the input model imaged after the LoSiTo run.
To create the empty MS file you have to run `lositosynthms_runner.py` and change the parameters in the `synthms.parset`, or, if you want you can run synthms as command line from the terminal after installing it.
Remember to put the MS file in the `/mss` folder.