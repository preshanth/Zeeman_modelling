import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Zeeman analysis and fitting")
parser.add_argument(
    "filename",
    type=str,
    nargs="+",
    help="Filename of the Stoeks I and Stokes V FITS file to be analysed",
)
parser.add_argument(
    "pixel", type=int, nargs=2, help="Pixel coordinates of the region to be analysed"
)
parser.add_argument(
    "--mapping",
    type=int,
    nargs="+",
    help="Number of Gaussian components to fit each visible peak",
)
parser.add_argument("--output", type=str, default=None, help="Output directory")
parser.add_argument("--justI", action="store_true", help="Only fits Stokes I")
parser.add_argument(
    "--vel",
    action="store_true",
    help="Plot x axis in LSR velocity. Otherwise defaults to frequency",
)
parser.add_argument(
    "--init", action="store_true", help="Visualize the position of initial guesses"
)
parser.add_argument("--trace", action="store_true", help="Plot trace plots")
parser.add_argument("--corner", action="store_true", help="Plot corner plots")
args = parser.parse_args()


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import arviz as az

az.style.use("arviz-darkgrid")

if not args.justI and len(args.filename) < 2:
    raise ValueError(
        "Stokes V FITS file not provided. Use --justI to fit Stokes I only"
    )


def main():
    # Read information from FITS files
    I_hdu = fits.open(args.filename[0])
    I_d = np.squeeze(I_hdu[0].data)
    I_h = I_hdu[0].header
    I_hdu.close()

    I = I_d[:, args.pixel[0], args.pixel[1]]

    d_nu = I_h["CDELT3"]
    nu_init = I_h["CRVAL3"]
    x_axis = nu_init + d_nu * np.arange(len(I))
    name = I_h["OBJECT"]

    from zeeman_modeling.func import mkdir

    cwd = mkdir(name, args.output)

    out_file = open(cwd + "output.txt", "w")

    from zeeman_modeling.func import guess_gen

    guess = guess_gen(I, args.mapping)
    num = len(guess) // 3
    noise_I = np.std(
        np.concatenate(
            (I[: int(guess[1] - 6 * guess[2])], I[int(guess[-2] + 6 * guess[-1]) :])
        ),
        ddof=1,
    )

    if args.init:
        plt.plot(I)
        for i in range(num):
            plt.axvline(guess[3 * i + 1], color="r")
        plt.savefig(cwd + "init_guess.png")

    from zeeman_modeling.bayesian import fit_I

    Itrace = fit_I(guess, I, noise_I)

    if args.trace:
        axes = az.plot_trace(Itrace, compact=False)
        fig = axes.ravel()[0].figure
        fig.savefig(cwd + "I_trace.png")

    if args.corner:
        import corner

        corner.corner(
            Itrace, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4e"
        ).savefig(cwd + "I_corner.png")

    post = Itrace.posterior
    mean = post.mean(dim=["chain", "draw"])

    from zeeman_modeling.plot import plotI

    amp = mean.amp.values
    mu = mean.mu.values
    sig = mean.sigma.values
    plotI(name, x_axis, I, amp, mu, sig, noise_I, out_file, cwd)

    if args.justI:
        out_file.close()
        return

    V_hdu = fits.open(args.filename[1])
    V_d = np.squeeze(V_hdu[0].data)
    V_h = V_hdu[0].header
    V_hdu.close()
    V = V_d[:, args.pixel[0], args.pixel[1]]

    noise_V = np.std(
        np.concatenate(
            (V[: int(guess[1] - 6 * guess[2])], V[int(guess[-2] + 6 * guess[-1]) :])
        ),
        ddof=1,
    )

    xs = np.arange(len(I))
    components = np.array(
        [amp[i] * np.exp(-0.5 * ((xs - mu[i]) / sig[i]) ** 2) for i in range(num)]
    )
    I_fit = np.sum(components, axis=0)
    from zeeman_modeling.bayesian import fit_V

    V_trace = fit_V(I_fit, V, d_nu, amp, mu, sig, noise_V)

    if args.trace:
        axes = az.plot_trace(V_trace, compact=False)
        fig = axes.ravel()[0].figure
        fig.savefig(cwd + "V_trace.png")

    if args.corner:
        import corner

        corner.corner(
            V_trace, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4e"
        ).savefig(cwd + "V_corner.png")

    post = V_trace.posterior
    mean = post.mean(dim=["chain", "draw"])
    alpha = mean.alpha.values
    beta = mean.beta.values

    from zeeman_modeling.plot import plotV

    plotV(name, x_axis, V, components, d_nu, alpha, beta, noise_V, out_file, cwd)

    out_file.close()
    from zeeman_modeling.plot import plot4pan

    plot4pan(
        name,
        x_axis,
        I,
        V,
        components,
        d_nu,
        amp,
        mu,
        sig,
        alpha,
        beta,
        noise_I,
        noise_V,
        cwd,
    )
