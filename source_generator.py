#!/usr/bin/env python

########################################
# Author: Matteo Cianfaglione          #
# e-mail: matteo.cianfaglione@unibo.it #
########################################

import os
import logging
import numpy as np
from astropy.io import fits
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



class checks:
    def __init__(self, x, y, z, imsize, image, r, exp, **kwargs):
        self.x = x
        self.y = y
        self.z = z
        self.imsize = imsize
        self.image = image
        self.r = r
        self.exp = exp
        self.__dict__.update(kwargs)

    def show_image(self, cmap = 'Blues', save = False):
        '''
        Shows the image of the sphere projection
        '''
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111)
        ax.tick_params('both', labelsize = 12)
        ax.set_xlabel('x [px]', fontsize = 16)
        ax.set_ylabel('y [px]', fontsize = 16)
        im = ax.imshow(self.image, cmap = cmap, origin = 'lower')
        clb = fig.colorbar(im, shrink = 0.8, aspect = 30, orientation = 'vertical')
        clb.set_label(label = 'Flux [Jy]', size = 18)
        clb.ax.tick_params(labelsize = 12)
        if save:
            plt.savefig('projection_2d.png', dpi = 300, bbox_inches = 'tight')
        plt.close()
        
    def plot3d(self, color = 'royalblue', s = 1, save = False):
        '''
        Shows the 3D plot of the sphere
        '''
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111, projection = '3d')
        # Set the plot limits
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

        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111)
        ax.tick_params('both', labelsize = 12)
        ax.set_xlabel(r'$r$ [pixel]', fontsize = 16)
        ax.set_ylabel('Density [sources/bin]', fontsize = 16)
        # plt.bar(bin_centers, densities, width = np.diff(bin_edges), color = c)
        ax.hist(bin_centers, facecolor = c, bins = bins, alpha = 0.75, edgecolor = 'black', weights = densities, density = False, label = 'Observed density')
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
        ax = fig.add_subplot(111)
        ax.tick_params('both', labelsize = 12)
        ax.set_xlabel('x [px]', fontsize = 16)
        ax.set_ylabel('y [px]', fontsize = 16)
        im = ax.imshow(self.exp, cmap = cmap, origin = 'lower', norm = 'log', vmin = 1e-8, vmax = 5e-5)
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
        fig, ax = plt.subplots(figsize = (8, 8))
        ax.tick_params('both', labelsize = 12)
        ax.set_xlabel('x [px]', fontsize = 16)
        ax.set_ylabel('y [px]', fontsize = 16)
        im = ax.imshow(self.image + self.exp, cmap = 'cubehelix', origin = 'lower', norm = 'log', vmin = 1e-8, vmax = 5e-5)
        clb = fig.colorbar(im, shrink = 0.8, aspect = 30, orientation = 'vertical')
        clb.set_label(label = 'Flux [Jy]', size = 18)
        clb.ax.tick_params(labelsize = 12)
        if save:
            plt.savefig('sources_and_exp.png', dpi = 300, bbox_inches = 'tight')
        plt.close()


class run:
    def __init__(self, n_points, r, I0, re, imsize, flux_value):
        self.n_points = n_points
        self.r = r
        self.I0 = I0
        self.re = re
        self.imsize = imsize
        self.flux_value = flux_value
        self.center = np.array([imsize / 2, imsize / 2])  # (x,y) coordinates of the center of the image
        
    def __call__(self):
        x, y, z = self.generate_uniform_sphere()
        logger.info(f"Generated {self.n_points} points in a sphere of radius {self.r:.1f} pixels.")
        image = self.generate_image(x, y)
        logger.info(f'Projected {self.n_points} points onto a 2D image of size {self.imsize}x{self.imsize} pixels.')
        logger.info(f'Flux value for each point: {self.flux_value} Jy.')
        return x, y, z, image
    
    def generate_uniform_sphere(self):
        ''' 
        Generates a uniform distribution of points in the volume of a sphere
        '''
        theta = np.arccos(2 * np.random.uniform(0, 1, self.n_points) - 1)
        phi = 2 * np.pi * np.random.uniform(0, 1, self.n_points)
        rho = self.r * np.cbrt(np.random.uniform(0, 1, self.n_points))  # Scale to the radius of the sphere
        # Convert spherical coordinates to Cartesian coordinates
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        return x, y, z
    
    def generate_exponential(self, pixsize):
        '''
        Generates a continuum distribution following a 2D exponential profile
        '''
        X = np.linspace(0, self.imsize, self.imsize)
        x, y = np.meshgrid(X, X)
        radius = np.sqrt((x - self.center[1])**2 + (y - self.center[0])**2)
        I = self.I0 * np.exp(-radius/self.re)
        logger.info(f'Generated an exponential profile with I0 = {self.I0/pixsize**2} uJy/arcsec^2 and re = {self.re:.1f} pixels.')
        return I

    def generate_image(self, x, y):
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
        x_pixel_valid = x_pixel[valid_mask]
        y_pixel_valid = y_pixel[valid_mask]
        
        # Add flux to the image at valid pixel locations.
        # np.add.at handles cases where multiple points map to the same pixel by summing.
        np.add.at(image, (y_pixel_valid, x_pixel_valid), self.flux_value)
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
        re = self.re * pixsize  # Convert re to arcsec
        I0 = self.I0            # Central brightness in Jy/arcsec^2
        flux = 2 * np.pi * I0/pixsize**2 * re**2 * f
        return flux
        

#####################
#    Main script    #
#####################

logger.info("Starting source generation...")

# Set up directories
dir_work = os.getcwd()
dir_plots = os.path.join(dir_work, 'plots')
dir_img = os.path.join(dir_work, 'img')
dir_parsets = os.path.join(dir_work, 'parsets')

directory(dir_plots)  # create directory for plots
directory(dir_img)    # create directory for images

logger.info(f"Working directory: {dir_work}")
logger.info(f"Image directory: {dir_img}")
logger.info(f"Parset directory: {dir_parsets}")
logger.info(f"Plots directory: {dir_plots}")

parset = os.path.join(dir_parsets, 'intact.parset')
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
name = variables['name']
n_points = int(variables['npoints'])   # Number of points to generate
r = float(variables['r'])              # Radius of the sphere in kpc
I0 = float(variables['I0'])            # Central brightness in Jy/arcsec^2
re = float(variables['re'])            # Effective radius in kpc
imsize = int(variables['imsize'])      # Image size in pixels
flux_value = float(variables['flux'])  # Flux value in Jy
scale = float(variables['scale'])      # Conversion scale in kpc/"
save = bool(variables['save'])
save_exp = bool(variables['save_exp'])
outname = variables['output']

# opens the fits file to get the header and pixsize
filename = dir_img + '/' + name + '.fits'
try:
    with fits.open(filename) as hdul:
        header = hdul[0].header
        pixsize = abs(header['CDELT2']) * 3600  # from deg to arcsec
except FileNotFoundError:
    logger.error(f"Input FITS file not found at {filename}")
except KeyError:
    logger.error(f"CDELT2 not found in FITS header of {filename}")


# Generate the sphere
r = r / scale / pixsize                   # Convert radius to pixels 
re = re / scale / pixsize                 # Effective radius in pixels
I0 = I0 * pixsize**2                      # Convert I0 to Jy/pixel for the model
sphere = run(n_points, r, I0, re, imsize, flux_value)
x, y, z, image = sphere()
fluxes = sphere.flux_calc(image)          # Estimate the total flux in the image
logger.info(f"Total flux in the image: {fluxes*1e3:.0f} mJy")

exp2d = sphere.generate_exponential(pixsize)  # Generate the exponential profile
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


# Check the distribution of points
os.chdir(dir_plots)
c = checks(x, y, z, imsize, image, r, exp2d)
c.show_image(save = save)
if n_points < 10000:
    c.plot3d(save = save)
c.show_dist(save = save)
c.exp2d_image(save = save)
c.sources_and_exp(save = save)  # Show the sources and the exponential profile
os.chdir(dir_work)


# Save the exponential profile if required
os.chdir(dir_img)
if save_exp:
    output_exp = f'exponential-model.fits'
    hdu_exp = fits.PrimaryHDU(exp2d, header)
    try:
        hdu_exp.writeto(output_exp, overwrite = True)
        logger.info(f"Exponential profile saved as {output_exp}")
    except Exception as e:
        logger.error(f"Error saving FITS file {output_exp}: {e}")
os.chdir(dir_work)        

logger.info("Source generation completed successfully.")