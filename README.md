# INTACT (INjector of mock discreTe sources in gAlaxy ClusTers)

INTACT (INjector of mock discreTe sources in gAlaxy ClusTers) - Pipeline to create a mock observation starting from a real one and injecting discrete sources in it. This is specifically thought for galaxy clusters
The current requirements are:
```
numpy
astropy
matplotlib
losito (and dependences therein)
```
This pipeline can be run inside the `flocs` or `PiLL` containers if you want to avoid installing the libraries on your machine.
This pipeline works by generating an empty LOFAR image with only noise in it using LoSiTo and then injecting in its MODEL_DATA column a uniform 3d distribution of sources (then projected into the 2d plane). Then the image can be produced with WSClean or other softwares for radio imaging.
The parameters are inputted via a file called `intact.parset`, there is an example file in the repo. It is important to notice that the imsize for the model must be the same as the input model imaged after the LoSiTo run.
To create the empty MS file you have to run `lositosynthms_runner.py` and change the parameters in the `synthms.parset`, or, if you want you can run synthms as command line from the terminal after installing it.
Remember to put the MS file in the `/mss` folder.

## Output

If you choose to save the output of `source_generator.py` you will find the plots saved in the directory `/plots`.
Some parameters of these plots can be changed from the script itself (colors, color maps, display the theoretical relation for the histogram).
Some examples of the plots are shown below.
![2D projection of the 3D uniform distribution of points in the sphere.](https://github.com/matteocianf/INTACT/blob/main/examples/sphere_projection.png)
![3D distribution of points inside the sphere](https://github.com/matteocianf/INTACT/blob/main/examples/sphere_3d.png)
![Histogram of the density of points in function of the radius of the sphere](https://github.com/matteocianf/INTACT/blob/main/examples/sphere_density_2d.png)

## Parsets

The pipeline works through the use of parset files.

The settings for `intact.parset` are:
+ `npoints`, which is the number of point sources to generate;
+ `r`, radius of the sphere over which you want to distribute the sources, must be expressed in kpc;
+ `imsize`, size of the image in pixels, must be equal to the size of the empty image;
+ `flux`, flux, in Jy, of each point source;
+ `save`, option to save the plots;
+ `name`, name of the fits file to open (the one from which you take the header - or the one in which you inject this model);
+ `scale`, scale in kpc/" for the object to simulate, this depends on the redshift;
+ `output`, name of the output model.

For `synthms.parset` the settings are:
+ `name`, name of the output MS file;
+ `tobs`, observation time in hours;
+ `station`, LOFAR station to use, can be `HBA`, `LBA` or `both`;
+ `minfreq`, lowest frequency of the observation in MHz (the code will convert in Hz);
+ `maxfreq`, highest frequency of the observation in MHz (the code will convert in Hz);
+ `lofarversion`, `1` for current version of LOFAR, `2` for the upgraded one;
+ `ra`, RA of the target in deg (the code will convert in rad);
+ `dec`, Dec of the target in deg (the code will convert in rad).
