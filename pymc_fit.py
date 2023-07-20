import pymc as pm
import numpy as np
from scipy.signal import find_peaks


class PyMCFItter:
    spectrum = None
    nGauss = 3

    def __init__(self, spectrum, nGauss) -> None:
        self.spectrum = spectrum # Spectrum to fit
        self.nGauss = nGauss # Number of Gaussians to fit per peak

    def guess_gen(self):
        guess = []
        peaks, info = find_peaks(self.spectrum, height=0., prominence=0.2, width = 0)
        for i in range(len(peaks)):
            center = peaks[i]
            width = info['widths'][i] * 0.6
            for j in np.linspace(center - width, center + width, self.nGauss):
                j += 5 # Horizontal offset to the right
                guess.append(self.spectrum[int(j)])
                guess.append(j)
                guess.append(width / self.nGauss)
        return guess    

    def fit_stokesI(self, tune = 10_000, draws = 10_000, cores = 4, chains = 4):
        xs = np.arange(len(self.spectrum))
        basic_model = pm.Model()
        guess = self.guess_gen()
        with basic_model as model:
            xdata = pm.ConstantData("x", xs)
            num = self.nGauss
            spec = self.spectrum
            # Priors for unknown model parameters
            amp = pm.Uniform("amp", lower=0, upper=np.max(self.spectrum) + 10, shape = num)
            mu = pm.Normal("mu", mu=guess[1::3], sigma=10, shape = num, transform=pm.distributions.transforms.univariate_ordered)
            sigma = pm.HalfNormal("sigma", sigma=10, shape=num)

            gauss_sum = np.sum([amp[i] * pm.math.exp(-0.5 * ((xdata - mu[i]) / sigma[i]) ** 2) for i in range(num)], axis=0)
            # Likelihood (sampling distribution) of observations
            likelihood = pm.Normal("y", mu=gauss_sum, observed=spec, sigma=0.01)

            # inference
            trace = pm.sample(draws=draws, tune=tune, cores = cores, chains = chains, discard_tuned_samples=False, step=pm.NUTS())
        return trace