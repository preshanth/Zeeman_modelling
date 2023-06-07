import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def nGauss(x, *params):
    """
    Function to fit the spectrum with a sum of Gaussian functions
    """
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        ctr = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -0.5 * ((x - ctr)/wid)**2)
    return y
    
class StokesIFit:
    spectrum = []
    sensitivity = 0.1
    popt = []
    pcov = []
    fit = []
    def __init__(self, spectrum, sensitiviy) -> None:
        self.spectrum = spectrum # Spectrum to fit
        self.sensitivity = sensitiviy # Acceptable signal peak as a fraction of the maximum signal


    def fit(self):
        """
        Fit the spectrum with a sum of Gaussian functions
        """
        # Find peaks in the spectrum
        peaks, info = find_peaks(self.spectrum, height=self.sensitivity*np.max(self.spectrum), width=0)
        # Initial guess for the parameters
        guess = []
        for i in range(len(peaks)):
            guess.append(self.spectrum[peaks[i]])
            guess.append(peaks[i] / len(self.spectrum))
            guess.append(info['widths'][i] / len(self.spectrum))
        # Fit the spectrum
        self.popt, self.pcov = curve_fit(nGauss, np.linspace(0, 1, len(self.spectrum)), self.spectrum, p0=guess)
        fit = nGauss(np.linspace(0, 1, len(self.spectrum)), *self.popt)
        return self.popt, self.pcov, fit