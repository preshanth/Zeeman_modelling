def mkdir(name: str, usr: str) -> str:
    """Function to make directory for output files depending on the user input.

    If the user input is None, then the directory will be named after the object name.
    If the user input is not None, then the directory will be named after the user input.

    Parameters
    ----------
        name: str
            Name of the observed object, read from the FITS file header.
        usr: str
            Name of the user defiend directory passed from args.output.

    Returns
    -------
        cwd: str
            Current working directory of the rest of the program.
    """

    from pathlib import Path

    if usr == None:
        Path(name).mkdir(parents=True, exist_ok=True)
        cwd = name + "/"

    else:
        Path(usr).mkdir(parents=True, exist_ok=True)
        cwd = usr + "/"

    return cwd

def guess_gen(spec, n_dist):
    import numpy as np
    from scipy.signal import find_peaks
    """Function to generate initial guesses based on spectral input and Gaussian components mapping.

    Generates initial guesses for the fitting process based on the spectral input and
    the number of Gaussian components to fit each visible peak.

    Parameters
    ----------
        spec: array_like
            Spectral input to be analysed.
        n_dist: array_like, optional
            Number of Gaussian components to fit each visible peak.

    Returns
    -------
        guess: array_like
            Initial guesses for the fitting process.
    """
    guess = []
    peaks, info = find_peaks(spec, height=0.0, prominence=0.2, width=0)

    if n_dist == None or len(peaks) != len(n_dist):
        n_dist = [] if n_dist == None else n_dist
        print(
            f"Mapping doesn't match. There are {len(peaks)} visible peaks but mapping for {len(n_dist)} peaks were provided. Defaults to 1 instead."
        )
        n_dist = np.ones(len(peaks)) * 1

    for i in range(len(peaks)):
        center = peaks[i]
        width = info["widths"][i] * 0.6
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