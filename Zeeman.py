import argparse
parser = argparse.ArgumentParser(description='Zeeman analysis and fitting')
parser.add_argument('filename', type=str, nargs=2, help='Filename of the Stoeks I and Stokes V FITS file to be analysed')
parser.add_argument('pixel', type=int, nargs=2, help='Pixel coordinates of the region to be analysed')
parser.add_argument('--mapping', type=int, nargs='+', help='Number of Gaussian components to fit each visible peak')
parser.add_argument('--output', type=str, default=None, help='Output directory')
parser.add_argument('--init', action='store_true', help='Visualize the position of initial guesses')
parser.add_argument('--plot', action='store_true', help='Plot the results')
# parser.add_argument('--verbose', action='store_true', help='Print out the results')
parser.add_argument('--trace', action='store_true', help='Plot trace plots')
parser.add_argument('--corner', action='store_true', help='Plot corner plots')
args = parser.parse_args()
# print(args)

def guess_gen(spec, n_dist):
    guess = []
    peaks, info = find_peaks(spec, height=0., prominence=0.2, width = 0)
    if n_dist == None or len(peaks) != len(n_dist): 
        print("Mapping doesn't match. Defaults to 1 instead.")
        n_dist = np.ones(len(peaks))*1
    for i in range(len(peaks)):
        center = peaks[i]
        width = info['widths'][i] * 0.6
        n = n_dist[i]
        if n == 1: 
            guess.append(spec[int(center)])
            guess.append(center)
            guess.append(width / n)
            continue
        for j in np.linspace(center - width, center + width, n):
            guess.append(spec[int(j)])
            guess.append(j)
            guess.append(width / n)

    return guess

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks

# Read information from FITS files
I_hdu = fits.open(args.filename[0])
I_d = np.squeeze(I_hdu[0].data)
I_h = I_hdu[0].header
I_hdu.close()

V_hdu = fits.open(args.filename[1])
V_d = np.squeeze(V_hdu[0].data)
V_h = V_hdu[0].header
V_hdu.close()

I = I_d[:, args.pixel[0], args.pixel[1]]
V = V_d[:, args.pixel[0], args.pixel[1]]
d_nu = I_h['CDELT3']
nu_init = I_h['CRVAL3']
x_axis = (nu_init + d_nu * np.arange(len(I))) / 1e9
name = I_h['OBJECT']
# ref_vel = I_h['ALTRVAL']
# ref_pix = I_h['ALTRPIX']
# vel_axis = (ref_vel + (np.arange(len(spec)) - ref_pix + 2) * (FreqtoLSR(nu_init) - FreqtoLSR(nu_init + d_nu))) / 1e3

# Make directory if not exist
from pathlib import Path
if args.output == None: 
    Path(name).mkdir(parents=True, exist_ok=True)
    cwd = name + '/'
else: 
    Path(args.output).mkdir(parents=True, exist_ok=True)
    cwd = args.output + '/'

guess = guess_gen(I, args.mapping)
num = len(guess) // 3
noise = np.std(np.concatenate((I[:int(guess[1] - guess[2])], I[int(guess[-2] + guess[-1]):])), ddof = 1)

if(args.init):
    plt.plot(I)
    for i in range(num):
        plt.axvline(guess[3 * i + 1], color='r')
    plt.savefig(cwd + 'init_guess.png')

import pymc as pm
I_model = pm.Model()
xs = np.arange(len(I))
with I_model as model:
    xdata = pm.ConstantData("x", xs)
    # Priors for unknown model parameters
    amp = pm.Uniform("amp", lower=0, upper=np.max(I)*1.2, shape = num)
    mu = pm.Normal("mu", mu=guess[1::3], sigma=10, shape = num, transform=pm.distributions.transforms.univariate_ordered)
    sigma = pm.HalfNormal("sigma", sigma=10, shape=num)

    gauss_sum = np.sum([amp[i] * pm.math.exp(-0.5 * ((xdata - mu[i]) / sigma[i]) ** 2) for i in range(num)], axis=0)
    # Likelihood (sampling distribution) of observations
    likelihood = pm.Normal("y", mu=gauss_sum, observed=I, sigma=noise)

    # inference
    trace = pm.sample(draws=10_000, tune=5_000, cores = 24, chains = 24, discard_tuned_samples=False, step = pm.NUTS(),
                      initvals={'amp': guess[0::3], 'mu': guess[1::3], 'sigma': guess[2::3]})

post = trace.posterior
mean = post.mean(dim=["chain", "draw"])
amp = mean.amp.values
mu = mean.mu.values
sig = mean.sigma.values
model = np.zeros(len(I))
x = np.arange(len(I))
for i in range(num):
    model += amp[i] * np.exp(-((x - mu[i]) ** 2) / (2 * sig[i] ** 2))

fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
axs[0].set(title=name + " Stokes I Fit", ylabel = "Flux Density (Jy)")
axs[0].plot(x_axis, I, 'o', color = 'green', label = 'Data', markersize = 4)
axs[0].plot(x_axis, model, label = 'Fit', color = 'c')
axs[0].plot(x_axis, I - model, 'x', markersize = 3, color = 'red', label = 'Residuals')
axs[0].errorbar(x_axis, I - model, yerr = 0.01, fmt = 'none', ecolor = 'k', elinewidth = 1, capsize = 2, alpha = 0.2)

for i in range(num):
    axs[0].plot(x_axis, amp[i] * np.exp(-((x - mu[i]) ** 2) / (2 * sig[i] ** 2)), label = f'Gauss{i}', alpha = 0.5)
    print('Amp:' , f"{amp[i]:.2f}", 'Center:', f"{mu[i]:.2f}", 'Width:', f"{sig[i]:.2f}", sep='\t') # Print to file

axs[0].legend()
axs[1].plot(x_axis, (I - model), label = 'Residuals')
axs[1].set(xlabel = "Frequency (GHz)")
axs[1].legend()
print('Chi2:', np.sum((I - model)**2))
fig.savefig(cwd + 'I_fit.png')
I_fit = model
