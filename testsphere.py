#!/usr/bin/env python

import os
import argparse
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


#########################
# Functions and classes #
#########################

def directory(directory):
   if not os.path.exists(directory):
     os.mkdir(directory)
     print(f"Directory {directory} created ")
   else:
     print(f"Directory {directory} already existing")


class checks:
    def __init__(self, x, y, z, imsize, image, r, **kwargs):
        self.x = x
        self.y = y
        self.z = z
        self.imsize = imsize
        self.image = image
        self.r = r
        self.__dict__.update(kwargs)

    def show_image(self, cmap = 'Blues', save = False):
        '''
        Shows the image of the sphere projection
        '''
        plt.figure(figsize = (8, 8))
        plt.imshow(image, cmap = cmap, origin = 'lower')
        plt.colorbar(label = 'Flux (Jy)')
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')
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
        ax.set_xlim([-limits, limits])
        ax.set_ylim([-limits, limits])
        ax.set_zlim([-limits, limits])
        ax.set_xlabel('x (pixel)')
        ax.set_ylabel('y (pixel)')
        ax.set_zlabel('z (pixel)')
        ax.set_aspect('equal', 'box')
        ax.scatter(self.x, self.y, self.z, s = s, c = color)
        if save:
            plt.savefig('sphere_3d.png', dpi = 300, bbox_inches = 'tight')
        plt.close()
        
    def show_dist(self, c = 'royalblue', bins = 50, save = False):
        '''
        Shows the histogram of the density of points in the prpjected image
        '''
        distances = np.sqrt(self.x**2 + self.y**2)  # distances from the center
        counts, bin_edges = np.histogram(distances, bins = bins, range = (0, self.r))

        # Calculate the area of each circular shell in 2D
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        shell_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
        # Calculate the density of points in each bin
        densities = counts / shell_areas

        theoretical_densities = np.sqrt(self.r**2 - bin_centers**2)
        theoretical_densities /= np.max(theoretical_densities)  # Normalize to the same scale
        theoretical_densities *= np.max(densities)  # Scale to match the observed densities
        
        plt.figure(figsize = (8, 8))
        plt.xlabel('r (pixel)')
        plt.bar(bin_centers, densities, width = np.diff(bin_edges), color = c)
        plt.plot(bin_centers, theoretical_densities, 'r-', label = 'Theoretical Density')        
        if save:
            plt.savefig('sphere_density_2d.png', dpi = 300, bbox_inches = 'tight')
        plt.close()


class run:
    def __init__(self, n_points, r, imsize, flux_value):
        self.n_points = n_points
        self.r = r
        self.imsize = imsize
        self.flux_value = flux_value
        self.center = np.array([imsize // 2, imsize // 2])  # (x,y) coordinates of the center of the image
        
    def __call__(self):
        x, y, z = self.generate_uniform_sphere()
        image = self.generate_image(x, y)
        return x, y, z, image
    
    def generate_uniform_sphere(self):
        ''' 
        Generates a uniform distribution of points in the volume of a sphere
        '''
        theta = np.arccos(2 * np.random.uniform(0, 1, self.n_points) - 1)
        phi = 2 * np.pi * np.random.uniform(0, 1, self.n_points)
        # rho = self.r * np.random.uniform(0, 1, self.n_points)  # strong overdensity in the center
        rho = self.r * np.cbrt(np.random.uniform(0, 1, self.n_points))
        
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        return x, y, z

    def generate_image(self, x, y):
        '''
        Generates a 2D image from the 3D points projected onto a 2D plane
        '''
        image = np.zeros((self.imsize, self.imsize))
        # Convert the 2D coordinates to pixel indices
        x_pixel = (x + self.center[0]).astype(int)
        y_pixel = (y + self.center[1]).astype(int)
        # Ensure the pixel indices are within the image bounds
        valid_indices = (x_pixel >= 0) & (x_pixel < imsize) & (y_pixel >= 0) & (y_pixel < imsize)
        # Assign the flux value to the corresponding pixels in the image
        for i in range(n_points):
            if valid_indices[i]:
                image[y_pixel[i], x_pixel[i]] += flux_value
        return image
        
    def flux_calc(self, image):
        '''
        Calculates the total flux in the image
        '''
        total_flux = np.sum(image)
        return total_flux


#####################
# Main script
#####################

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('-npoints', dest = 'npoints', nargs = '?', type = int, const = 1, help = 'Number of points to generate')
parser.add_argument('-r', dest = 'r', nargs = '?', type = float, const = 1, help = 'Radius of the sphere')
parser.add_argument('-imsize', dest = 'imsize', nargs = '?', type = int, const = 1, help = 'Image size in pixels')
parser.add_argument('-flux', dest = 'flux', nargs = '?', type = float, const = 1, help = 'Flux value in Jy of each source')
parser.add_argument('-check', dest = 'check', action = 'store_true', help = 'Check the image and the distribution of points')
parser.add_argument('-save', dest = 'save', action = 'store_true', help = 'Save the image and the distribution of points')
parser.add_argument('-n', dest = 'name', nargs = '?', type = str, help = 'Name of the image from which you take the header information')
parser.add_argument('-scale', nargs = '?', type = float, const = 1, help = 'Conversion scale in kpc/"')


args = parser.parse_args()
n_points = args.npoints   # Number of points to generate
r = args.r                # Radius of the sphere in kpc
imsize = args.imsize      # Image size in pixels
flux_value = args.flux    # Flux value in Jy
check = args.check
save = args.save
name = args.name
scale = args.scale


dir_work = os.getcwd() + '/'
dir_plots = dir_work + 'plots/'
dir_img = dir_work + 'img/'

directory(dir_plots)  # create directory for plots

filename = dir_work + f'{name}.fits'
hdul = fits.open(filename)
header = hdul[0].header
data = np.array(hdul[0].data[0,0,:,:]) # in this way I have a 2x2 array excluding the other axes
pixsize = abs(header['CDELT1']) * 3600
hdul.close()

r = r / scale / pixsize
sphere = run(n_points, r, imsize, flux_value, pixsize)
x, y, z, image = sphere()
fluxes = sphere.flux_calc(image)
print(f'Total flux: {fluxes} Jy')

output = dir_img + 'model_image.fits'
hdu = fits.PrimaryHDU(image, header)
hdu.writeto(output, overwrite = True)

if check:
    os.chdir(dir_plots)
    c = checks(x, y, z, imsize, image, r)
    c.show_image(save = save)
    if n_points < 10000:
        c.plot3d(save = save)
    c.show_dist(save = save)
    os.chdir(dir_work)