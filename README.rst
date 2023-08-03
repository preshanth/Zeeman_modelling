Zeeman_modelling
================

About
-----

This command line program written in Python aims to standardize and automize the Zeeman
anlaysis process in emission line radio astronomy through robust curve fitting and Bayesian-based
parameter sampling. We hope to minimize user bias in the study of
interstellar magnetic fields through this software solution.

Getting Started
---------------

- Clone this repository.
- It is highly to conduct the following installation in a clean environment. Could be conda, mamba, or venv.
- cd into the cloned folder and enter the following command:
.. code:: sh

   pip install .

- cd into the directory where the Stokes I and Stokes V cubes are and call the program by:
.. code:: sh
   
   zeeman_modeling FILENAME_STOKESI [FILENAME_STOKESV] XPIXEL YPIXEL [OPTIONS]

- see detailed options and examples below 
- To verify correct installation, run:
.. code:: sh

   zeeman_modeling -h 
   # or
   zeeman_modeling --help

- Expected output
.. code-block:: console

   usage: zeeman_modeling [-h] [--mapping MAPPING [MAPPING ...]] [--output OUTPUT] [--justI] [--vel] [--init] [--trace]
                       [--corner]
                       filename [filename ...] pixel pixel

   Zeeman analysis and fitting
      
   positional arguments:
     filename              Filename of the Stoeks I and Stokes V FITS file to be analysed
     pixel                 Pixel coordinates of the region to be analysed
   
   options:
     -h, --help            show this help message and exit
     --mapping MAPPING [MAPPING ...]
                           Number of Gaussian components to fit each visible peak
     --output OUTPUT       Output directory
     --justI               Only fits Stokes I
     --vel                 Plot x axis in LSR velocity. Otherwise defaults to frequency
     --init                Visualize the position of initial guesses
     --trace               Plot trace plots
     --corner              Plot corner plots


Usage
-----

**It is highly encouraged for users to see where the initial guesses by the
–init option before proceeding to Stokes I and Stokes V fitting.**
   
:filename_I:   filename of the Stokes I cube
:filename_V:   filename of the Stokes V cube, optional when using the --justI flag
:pixel_x:      x value of the selected pixel
:pixel_y:      y value of the selected pixel

-h, --help      show this help message and exit
--mapping       [MAPPING …] Number of Gaussian components to fit each visible peak
--output        Output directory 
--justI         Conduct line fitting on just the Stokes I profile
--init          Visualize the position of initial guesses 
--trace         Plot trace plots
--corner        Plot corner plots

Example
-------

Visualize initial guess:

.. code:: sh

   zeeman_modeling TestI.py TestV.py 64 64 --init

To fit Stokes I and Stokes V FITS cubes named 1720I.FITS and 1720V.FITS
at pixel value x = 128, y = 128, one would pass the following line to
the terminal. (Assuming there are two visible peaks in Stokes I profile
and would like to fit each peak with two Gaussian components)

.. code:: sh

   zeeman_modeling 1720_I.FITS 1720_V.FITS 128 127 --mapping 2 2

To also store the plotting results as well as corner plots and tracing
plots to check for MCMC effectiveness, add the following arguments:

.. code:: sh

   zeeman_modeling 1720_I.FITS 1720_V.FITS 128 127 --mapping 2 2 --plot --trace --corner
