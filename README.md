# Zeeman_modelling

## About

This Python scirpt file aims to standardize and automize the Zeeman anlaysis process through robust curve fitting and Bayesian-based parameter sampling. We hope to minimize user bias in the study of interstellar magnetic fields through this software solution.

## Getting Started
Below are a list of required Python packges:
- Numpy
- Scipy
- PyMC
- Matplotlib
- Astropy

After installing the required Python packages, simply put **BOTH** the Stokes I and Stokes V cubes into the same working directory as the Zeeman.py script file before executing the commands.

## Usage
_**It is highly encouraged for users to see where the initial guesses by --init before proceeding to Stokes I and Stokes V fitting.**_


Zeeman.py [-h] [--mapping MAPPING [MAPPING ...]] [--output OUTPUT] [--init] [--plot] [--trace] [--corner] filename_I filename_V pixel_x pixel_y

positional arguments:
- filename_I filename of the Stokes I cube
- filename_V filename of the Stokes V cube
- pixel_x x value of the selected pixel
- pixel_y y value of the selected pixel

options:
- -h, --help            show this help message and exit
- --mapping MAPPING [MAPPING ...] Number of Gaussian components to fit each visible peak
- --output OUTPUT       Output directory
- --init                Visualize the position of initial guesses
- --plot                Plot the results
- --trace               Plot trace plots
- --corner              Plot corner plots


## Example
Visualize initial guess:
```sh
python Zeeman.py TestI.py TestV.py 64 64 --init
```

To fit Stokes I and Stokes V FITS cubes named 1720I.FITS and 1720V.FITS at pixel value x = 128, y = 128, one would pass the following line to the terminal. (Assuming there are two visible peaks in Stokes I profile and would like to fit each peak with two Gaussian components)
```sh
python Zeeman.py 1720_I.FITS 1720_V.FITS 128 127 --mapping 2 2
```
To also store the plotting results as well as corner plots and tracing plots to check for MCMC effectiveness, add the following arguments:

```sh
python Zeeman.py 1720_I.FITS 1720_V.FITS 128 127 --mapping 2 2 --plot --trace --corner
```
