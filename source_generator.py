#!/usr/bin/env python

########################################
# Author: Matteo Cianfaglione          #
# e-mail: matteo.cianfaglione@unibo.it #
########################################

import os
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt


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

#########################
# Functions and classes #
#########################

def directory(directory):
   if not os.path.exists(directory):
      os.mkdir(directory)
      logger.info(f"Directory {directory} created")
   else:
      logger.info(f"Directory {directory} already exists")


class run:
   def __init__(self, n_points, r, I0, re, imsize, flux_hist, flux_bin_edges, dist, flux_list = None, ra = None, dec = None, wcs = None):
      self.n_points       = n_points
      self.r              = r
      self.I0             = I0
      self.re             = re
      self.imsize         = imsize
      self.flux_hist      = flux_hist
      self.flux_bin_edges = flux_bin_edges
      self.flux_list      = flux_list
      self.ra             = ra
      self.dec            = dec
      self.wcs            = wcs
      self.dist           = dist
      self.center         = np.array([imsize / 2, imsize / 2])  # (x,y) coordinates of the center of the image

   def __call__(self):
      x, y, z, flux_values = self.generate_sphere()
      logger.info(f"Generated {self.n_points} points in a sphere of radius {self.r:.1f} pixels.")
      image = self.generate_image(x, y, flux_values)
      logger.info(f'Projected {self.n_points} points onto a 2D image of size {self.imsize}x{self.imsize} pixels.')
      return x, y, z, image, flux_values
    
   def generate_sphere(self):
      ''' 
      Generates a uniform distribution of points in the volume of a sphere
      '''
      if self.ra is None and self.dec is None:
         if self.dist == 'sphere':
            logger.info("Generating points uniformly distributed within a sphere.")
            theta = np.arccos(2 * np.random.uniform(0, 1, self.n_points) - 1)
            phi   = 2 * np.pi * np.random.uniform(0, 1, self.n_points)
            rho   = self.r * np.cbrt(np.random.uniform(0, 1, self.n_points))  # Scale to the radius of the sphere
            # Convert spherical coordinates to Cartesian coordinates
            x = rho * np.sin(theta) * np.cos(phi)
            y = rho * np.sin(theta) * np.sin(phi)
            z = rho * np.cos(theta)
         elif self.dist == 'flat':
            logger.info("Generating points uniformly distributed within a flat disk.")
            theta = 2 * np.pi * np.random.uniform(0, 1, self.n_points)
            rho   = self.r * np.random.uniform(0, 1, self.n_points) ** 0.5
            x     = rho * np.cos(theta)
            y     = rho * np.sin(theta)
            z     = np.zeros_like(x)  # z-coordinates are zero for a flat disk
      else:
         if self.wcs is not None:
            pix_coords = self.wcs.all_world2pix(self.ra, self.dec, 0)
            x_pix, y_pix = pix_coords
         else:
            x_pix = self.ra
            y_pix = self.dec
         logger.info("Using provided RA and Dec coordinates for point generation.")
         
         x = x_pix - self.center[0]
         y = y_pix - self.center[1]
         z = np.zeros_like(x)  # z-coordinates are zero for a 2D projection
            
      if self.flux_list is not None:
         flux_values = self.flux_list
      else:
         flux_values = np.zeros(self.n_points)
         start_idx = 0
         for i in range(len(self.flux_bin_edges) - 1):
            num_points_in_bin = int(self.flux_hist[i])
            if num_points_in_bin > 0:
               # Generate random flux values within the current bin
               bin_flux_values = np.random.uniform(self.flux_bin_edges[i], self.flux_bin_edges[i + 1], num_points_in_bin)
               # Correctly place the generated fluxes into the main array
               flux_values[start_idx : start_idx + num_points_in_bin] = bin_flux_values
               start_idx += num_points_in_bin
      return x, y, z, flux_values
   
    
   def generate_exponential(self, pixsize):
      '''
      Generates a continuum distribution following a 2D exponential profile
      '''
      X      = np.linspace(0, self.imsize, self.imsize)
      x, y   = np.meshgrid(X, X)
      radius = np.sqrt((x - self.center[1])**2 + (y - self.center[0])**2)
      I      = self.I0 * np.exp(-radius/self.re)
      logger.info(f'Generated an exponential profile with I0 = {self.I0/pixsize**2} uJy/arcsec^2 and re = {self.re:.1f} pixels.')
      return I

   def generate_image(self, x, y, flux_values):
      '''
      Generates a 2D image from the 3D points projected onto a 2D plane
      '''
      image = np.zeros((self.imsize, self.imsize))
      # Convert the 2D coordinates to pixel indices
      x_pixel = (x + self.center[0]).astype(int)
      y_pixel = (y + self.center[1]).astype(int)
      
      valid_mask = (x_pixel >= 0) & (x_pixel < self.imsize) & \
               (y_pixel >= 0) & (y_pixel < self.imsize)
      # Filter the coordinates
      x_pixel_valid     = x_pixel[valid_mask]
      y_pixel_valid     = y_pixel[valid_mask]
      flux_values_valid = flux_values[valid_mask]
      
      # Add flux to the image at valid pixel locations.
      # np.add.at handles cases where multiple points map to the same pixel by summing.
      np.add.at(image, (y_pixel_valid, x_pixel_valid), flux_values_valid)
      return image
        
   def flux_calc(self, image):
      '''
      Computes the total flux in the image
      '''
      total_flux = np.sum(image)
      return total_flux
    
   def flux_exp(self, f = 0.8):
      '''
      Computes the flux of the exponential profile up to 3re (f = 0.8)
      I0 in Jy/arcsec^2, re in arcsec
      '''
      re   = self.re * pixsize  # Convert re to arcsec
      I0   = self.I0            # Central brightness in Jy/arcsec^2
      flux = 2 * np.pi * I0/pixsize**2 * re**2 * f
      return flux
        

class checks:
   def __init__(self, x, y, z, imsize, image, r, exp, flux_values, flux_histogram, flux_bin_edges, **kwargs):
      self.x              = x
      self.y              = y
      self.z              = z
      self.imsize         = imsize
      self.image          = image
      self.r              = r
      self.exp            = exp
      self.flux_values    = flux_values
      self.flux_hist      = flux_histogram
      self.flux_bin_edges = flux_bin_edges
      self.__dict__.update(kwargs)

   def show_image(self, cmap = 'cubehelix', save = False):
      '''
      Shows the image of the sphere projection
      '''
      fig = plt.figure(figsize = (8, 8))
      ax  = fig.add_subplot(111)
      ax.tick_params('both', labelsize = 12)
      ax.set_xlabel('x [px]', fontsize = 16)
      ax.set_ylabel('y [px]', fontsize = 16)
      im  = ax.imshow(self.image, cmap = cmap, origin = 'lower', vmin = 0, vmax = np.max(self.image))
      clb = fig.colorbar(im, shrink = 0.8, aspect = 30, orientation = 'vertical')
      clb.set_label(label = 'Flux [Jy]', size = 18)
      clb.ax.tick_params(labelsize = 12)
      if save:
         plt.savefig('projection_2d.png', dpi = 600, bbox_inches = 'tight')
      plt.close()
        
   def plot3d(self, color = 'royalblue', s = 1, save = False):
      '''
      Shows the 3D plot of the sphere
      '''
      fig = plt.figure(figsize = (8, 8))
      ax  = fig.add_subplot(111, projection = '3d')
      limits = self.imsize/2
      ax.tick_params('both', labelsize = 12)
      ax.set_xlim([-limits, limits])
      ax.set_ylim([-limits, limits])
      ax.set_zlim([-limits, limits])
      ax.set_xlabel('x [px]', fontsize = 16)
      ax.set_ylabel('y [px]', fontsize = 16)
      ax.set_zlabel('z [px]', fontsize = 16)
      ax.set_aspect('equal', 'box')
      ax.scatter(self.x, self.y, self.z, s = s, c = color)
      if save:
         plt.savefig('sphere_3d.png', dpi = 300, bbox_inches = 'tight')
      plt.close()

   def show_dist(self, c = 'royalblue', bins = 10, save = False):
      '''
      Shows the histogram of the density of points in the prpjected image
      '''
      distances = np.hypot(self.x, self.y)        # distances from the center
      counts, bin_edges = np.histogram(distances, bins = bins, range = (0, self.r))
      # Calculate the area of each circular shell in 2D
      bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
      shell_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
      # Calculate the density of points in each bin
      densities = counts / shell_areas
      theoretical_densities = np.sqrt(self.r**2 - bin_centers**2)
      theoretical_densities /= np.mean(theoretical_densities)      # Normalize to the same scale
      theoretical_densities *= np.mean(densities)                  # Scale to match the observed densities            
      # Plot the histogram
      fig = plt.figure(figsize = (8, 8))
      ax  = fig.add_subplot(111)
      ax.tick_params('both', labelsize = 12)
      ax.set_xlabel(r'$r$ [pixel]', fontsize = 16)
      ax.set_ylabel('Density [sources/bin]', fontsize = 16)
      # plt.bar(bin_centers, densities, width = np.diff(bin_edges), color = c)
      ax.hist(bin_centers, weights = densities, density = False, bins = bins, \
          facecolor = c, alpha = 0.75, edgecolor = 'black', label = 'Observed density')
      ax.plot(bin_centers, theoretical_densities, 'r-', label = 'Theoretical density')
      plt.legend(loc = 'best')        
      if save:
         plt.savefig('sphere_density_2d.png', dpi = 300, bbox_inches = 'tight')
      plt.close()

   def exp2d_image(self, cmap = 'cubehelix', save = False):
      '''
      Shows the 2D exponential profile
      '''
      fig = plt.figure(figsize = (8, 8))
      ax  = fig.add_subplot(111)
      ax.tick_params('both', labelsize = 12)
      ax.set_xlabel('x [px]', fontsize = 16)
      ax.set_ylabel('y [px]', fontsize = 16)
      im  = ax.imshow(self.exp, cmap = cmap, origin = 'lower', norm = 'log', vmin = 1e-8, vmax = 1e-4)
      clb = fig.colorbar(im, shrink = 0.8, aspect = 30, orientation = 'vertical')
      clb.set_label(label = 'Flux [Jy]', size = 18)
      clb.ax.tick_params(labelsize = 12)
      if save:
         plt.savefig('exponential_profile.png', dpi = 300, bbox_inches = 'tight')
      plt.close()
        
   def sources_and_exp(self, cmap = 'cubehelix', save = False):
      '''
      Shows the 2D image with the sources and the exponential profile
      '''
      img     = self.image + self.exp  # Combine the image and the exponential profile
      fig, ax = plt.subplots(figsize = (8, 8))
      ax.tick_params('both', labelsize = 12)
      ax.set_xlabel('x [px]', fontsize = 16)
      ax.set_ylabel('y [px]', fontsize = 16)
      im  = ax.imshow(img, cmap = cmap, origin = 'lower', norm = 'log', vmin = 1e-8, vmax = np.max(img))
      clb = fig.colorbar(im, shrink = 0.8, aspect = 30, orientation = 'vertical')
      clb.set_label(label = 'Flux [Jy]', size = 18)
      clb.ax.tick_params(labelsize = 12)
      if save:
         plt.savefig('sources_and_exp.png', dpi = 600, bbox_inches = 'tight')
      plt.close()
        
   def show_flux_histogram(self, save = False):
      '''
      Shows the histogram of the flux values
      '''
      fig = plt.figure(figsize = (8, 8))
      ax  = fig.add_subplot(111)
      ax.tick_params('both', labelsize = 12)
      ax.set_xlabel('Flux [Jy]', fontsize = 16)
      ax.set_ylabel('Number of sources', fontsize = 16)
      ax.set_xlim(0, 0.02)
      ax.hist(self.flux_values, bins=300, 
               color='red', edgecolor='k', alpha=0.7, label='Generated Distribution')
      ax.hist(self.flux_bin_edges[:-1], bins = self.flux_bin_edges, weights = self.flux_hist, \
               color = 'royalblue', edgecolor = 'k', alpha = 0.75, label = 'Target distribution')
      ax.legend()
      if save:
         plt.savefig('flux_histogram.png', dpi = 300, bbox_inches = 'tight')
      plt.close()

#####################
#    Main script    #
#####################

logger.info("Starting source generation...")

# Set up directories
dir_work    = os.getcwd()
dir_plots   = os.path.join(dir_work, 'plots')
dir_img     = os.path.join(dir_work, 'img')
dir_parsets = os.path.join(dir_work, 'parsets')

directory(dir_plots)  # create directory for plots
directory(dir_img)    # create directory for images

logger.info(f"Working directory: {dir_work}")
logger.info(f"Image directory: {dir_img}")
logger.info(f"Parset directory: {dir_parsets}")
logger.info(f"Plots directory: {dir_plots}")

parset    = os.path.join(dir_parsets, 'intact.parset')
variables = {}
try:
   with open(parset, 'r') as file:
      for line in file:
         line = line.strip()                        # Removes spaces and newlines
         if not line or line.startswith('#'):       # Skip empty lines or comments
            continue
         if '=' in line:
            key, value = line.split('=', 1)        # Splits on the first '='
            variables[key.strip()] = value.strip() # adds to the dictionary
except FileNotFoundError:
   logger.error(f"Parset file not found at {parset}")# Exit if parset file is missing
            
# Parameters
# REMEMBER IMSIZE MUST BE THE SAME AS THE INITIAL IMAGE
name     = variables['name']
r        = float(variables['r'])              # Radius of the sphere in kpc
I0       = float(variables['I0'])             # Central brightness in Jy/arcsec^2
re       = float(variables['re'])             # Effective radius in kpc
scale    = float(variables['scale'])          # Conversion scale in kpc/"
save     = int(variables['save'])
save_exp = int(variables['save_exp'])
outname  = variables['output']
list     = int(variables['list'])             # If 1 use the flux list
rms      = float(variables['rms']) * 1e-6     # RMS in Jy/beam
coord    = variables['coord_scale']           # pixels or deg
dist     = variables['distribution']          # sphere or flat
density  = float(variables['density'])        # sources per arcmin^2

# opens the fits file to get the header and pixsize
filename = dir_img + '/' + name + '.fits'
try:
   with fits.open(filename) as hdul:
      header  = hdul[0].header
      pixsize = abs(header['CDELT2']) * 3600  # from deg to arcsec
      imsize  = header['NAXIS1']               # Assuming square image, NAXIS1 == NAXIS2
      if coord == 'deg':
         wcs = WCS(header)
         wcs = wcs.dropaxis(-1).dropaxis(-1)
      else:
         wcs = None
except FileNotFoundError:
   logger.error(f"Input FITS file not found at {filename}")
except KeyError:
   logger.error(f"CDELT2 not found in FITS header of {filename}")

flux = None; ra = None; dec = None
if not list:
   logger.info("Generating flux histogram from scratch...")
   flux_histogram = np.array([150])
   flux_bin_edges = np.linspace(3, 3, len(flux_histogram) + 1) * rms 
   if density == 0:
      n_points = np.sum(flux_histogram)    # Total number of points to generate
   else:
      r_amin = r / scale / 60        # radius in arcmin
      area   = np.pi * r_amin**2     # area in arcmin^2
      n_points = int(density * area) # Total number of points to generate
      flux_histogram = np.array([n_points])
      flux_bin_edges = np.linspace(3, 3, len(flux_histogram) + 1) * rms 
   logger.info(f"Total number of points to generate: {n_points}")
   logger.info(f"The flux bins are: \n{flux_bin_edges*1e3} mJy")
else:
   logger.info("Using flux list from CSV file...")
   data = pd.read_csv('./flux_list.csv', header = 0, sep = ' ')
   flux = data['Flux'].values * rms
   n_points = len(flux)
   flux_histogram, flux_bin_edges = np.histogram(flux, bins = 50, range = (0, np.max(flux)), density = False)
   logger.info(f"Total number of points to generate: {n_points}")
   logger.info(f"The flux bins are: \n{flux_bin_edges} * 1e3 mJy")
    # Check if RA and Dec columns exist before reading them
   if 'RA' in data.columns and 'Dec' in data.columns:
      logger.info("RA and Dec columns found in CSV file.")
      ra  = data['RA'].values
      dec = data['Dec'].values
   else:
      logger.warning("RA and Dec columns not found in CSV. Positions will be generated randomly.")


# Generate the sphere
r  = r / scale / pixsize                   # Convert radius to pixels 
re = re / scale / pixsize                 # Effective radius in pixels
I0 = I0 * pixsize**2                      # Convert I0 to Jy/pixel for the model
sphere = run(n_points, r, I0, re, imsize, flux_histogram, flux_bin_edges, dist, flux_list = flux, ra = ra, dec = dec, wcs = wcs)
x, y, z, image, flux_values = sphere()
fluxes = sphere.flux_calc(image)          # Estimate the total flux in the image
logger.info(f"Total flux in the image: {fluxes*1e3} mJy")


exp2d    = sphere.generate_exponential(pixsize)  # Generate the exponential profile
flux_exp = sphere.flux_exp()                  # Estimate the flux of the exponential profile
logger.info(f"Flux of the exponential profile up to 3re: {flux_exp*1e3:.0f} mJy")

os.chdir(dir_img)
output = f'{outname}-model.fits'
hdu = fits.PrimaryHDU(image, header)
hdu.writeto(output, overwrite = True)
try:
   hdu.writeto(output, overwrite = True)
   logger.info(f"Image saved as {output}")
except Exception as e:
   logger.error(f"Error saving FITS file {output}: {e}")
os.chdir(dir_work)

# Save the exponential profile if required
os.chdir(dir_img)
if save_exp:
   output_exp = f'exponential-model.fits'
   hdu_exp    = fits.PrimaryHDU(exp2d, header)
   try:
      hdu_exp.writeto(output_exp, overwrite = True)
      logger.info(f"Exponential profile saved as {output_exp}")
   except Exception as e:
      logger.error(f"Error saving FITS file {output_exp}: {e}")
os.chdir(dir_work) 

# Check the distribution of points
os.chdir(dir_plots)
c = checks(x, y, z, imsize, image, r, exp2d, flux_values, flux_histogram, flux_bin_edges, flux_list = flux)
c.show_image(save = save)
if n_points < 10000:
   c.plot3d(save = save)
c.show_dist(save = save)
c.exp2d_image(save = save)
c.sources_and_exp(save = save)  # Show the sources and the exponential profile
c.show_flux_histogram(save = save)  # Show the histogram of flux values
os.chdir(dir_work)       

logger.info("Source generation completed successfully.")
