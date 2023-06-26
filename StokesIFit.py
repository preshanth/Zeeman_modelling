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
        peaks, info = find_peaks(self.spectrum, height=self.sensitivity*np.max(self.spectrum), width=0.01 * len(self.spectrum))
        if len(peaks) <= 0: raise Exception('No peaks found in the spectrum')
        # Initial guess for the parameters, assuems each peak is a sum of two symmetric Gaussians
        guess = []
        for i in range(len(peaks)):
            guess.append(self.spectrum[peaks[i]]) # Amplitude
            guess.append(peaks[i] + info['widths'][i] / 4) # Center
            guess.append(info['widths'][i] / 2) # Width

            guess.append(self.spectrum[peaks[i]]) # Amplitude
            guess.append(peaks[i] - info['widths'][i] / 4) # Center
            guess.append(info['widths'][i] / 2) # Width

        # Fit the spectrum
        self.popt, self.pcov = curve_fit(nGauss, np.arange(0, len(self.spectrum), 1.), self.spectrum, p0=guess)
        fit = nGauss(np.arange(0, len(self.spectrum), 1.), *self.popt)
        return self.popt, self.pcov, fit
    
    def lisParam(self):
        """
        List the parameters of the fit
        """
        if len(self.popt) <= 0: raise Exception('No fit parameters found, run fit() first')
        print('Fit parameters:')
        for i in range(0, len(self.popt), 3):
            print('Peak %d: Amplitude = %.2f, Center = %.2f, Width = %.2f' % (i/3, self.popt[i], self.popt[i+1], self.popt[i+2]))