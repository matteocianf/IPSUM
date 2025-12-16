# IPSUM (Injector of Point SoUrces in Mock observations)

INTACT (INjector of mock poinT sources in gAlaxy ClusTers) - Pipeline to create a mock observation starting from a real one and injecting discrete sources in it. This is specifically thought for galaxy clusters
The current requirements are:

```[text]
numpy
astropy
matplotlib
casacore
pandas
losito (and dependences therein)
wsclean
DP3
```

This pipeline can be run inside the [flocs](https://github.com/tikk3r/flocs) container if you want to avoid installing the libraries on your machine.
This pipeline works by generating an empty LOFAR image with only noise in it using [LoSiTo](https://github.com/darafferty/losito) and then injecting in the visibilities a distribution of sources. Then the image can be produced with WSClean or other softwares for radio imaging.
The distribution can be given through a CSV with also the coordinates of the points or randomly generated with a uniform flux distribution, the edges of the distribution must be specified and also the number of sources. (I'm working to make this easier as for now the parameters must be modifed inside the code instead of a parset)
The parameters for the different scripts are given via `parsets`, examples of these are found in the `parsets` directory.
Before running the injector you need to create an empty image with only noise (I'm working on automating this, too).

The first step is to run `losito_runner.py` to generate the `MS` file, by default it only adds noise in the visibilities, to do more complex operations I recommend to look at the wiki of LoSiTo and modify the parset based on it.
After this, create the image with only noise.
To create the sources model run `source_generator.py`.
To inject the models and create the images run `pred_inj.py`.

## Output

If you choose to save the output of `source_generator.py` you will find the plots saved in the directory `/plots`.
Some parameters of these plots can be changed from the script itself (colors, color maps, display the theoretical relation for the histogram).

## Parsets

The pipeline works through the use of parset files.

### losito_runner.py

For LoSiTo different parsets are used.

#### synthms.parset

The parameters here are:

+ `name`, sets the name of the output `MS` file, no need to add `.MS` in the name;
+ `tobs`, observation time in hours;
+ `station`, LOFAR station to use, can be `HBA`, `LBA` or `both`;
+ `minfreq`, lowest frequency of the observation in MHz (the code will convert in Hz);
+ `maxfreq`, highest frequency of the observation in MHz (the code will convert in Hz);
+ `lofarversion`, `1` for current version of LOFAR, `2` for LOFAR2.0;
+ `ra`, RA of the target in deg (the code will convert in rad);
+ `dec`, Dec of the target in deg (the code will convert in rad);
+ `chanpersb`, number of channels per sub-band;
+ `tres`, integration time, or time resolution of the MS, in seconds;
+ `start`, start time of the observation in MJD.

#### dp3_freqavg.parset

DP3 parset to average in frequency the initial `MS` file.

#### losito.parset

LoSiTo parset to add noise and corruptions to the `MS` file.
Look at LoSiTo documentation for more information.

### source_generator.py

#### ipsum.parset

Parset used for the creation of point sources model and radio halo model:

+ `r`, radius of the sphere over which you want to distribute the sources, must be expressed in kpc;
+ `save`, option to save the plots (either 1 (True) or 0 (False));
+ `name`, name of the fits image with only noise;
+ `scale`, scale in kpc/" for the object to simulate, this depends on the redshift;
+ `output`, name of the output model;
+ `I0`, central brightness of the exponential profile, expressed in $Jy/arcsec^2$;
+ `re`, effective radius of the exponential in kpc;
+ `save_exp`, to save as a `fits` image the exponential;
+ `list`, activate if you want to use a source flux list (either 1 (True) or 0 (False));
+ `coord`, select the type of coordinates for the sources.
+ `distribution`, choose between flat and sphere, both uniform distribution but one is uniform in a 3D sphere and one in a 2D disk;
+ `density`, you can specify the density of the points per arcmin^2 instead of letting the code generate a fixed number of points.

### pred_inj.py

#### inj.parset

To select the parameters for injection, source subtraction and imaging:

+ `mssname`, name of the `MS` file for imaging;
+ `name`, name of the cluster or target;
+ `only_sub`, to only do the subtraction step;
+ `minuv_sub`, to select the inner uv-cut.
