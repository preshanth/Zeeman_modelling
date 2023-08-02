import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Zeeman analysis and fitting")
parser.add_argument(
    "filename",
    type=str,
    nargs=2,
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
parser.add_argument(
    "--init", action="store_true", help="Visualize the position of initial guesses"
)
parser.add_argument(
    "--plotI", action="store_true", help="Plot individually the stokes I results"
)
parser.add_argument(
    "--plotV", action="store_true", help="Plot individually the stokes V results"
)
parser.add_argument("--trace", action="store_true", help="Plot trace plots")
parser.add_argument("--corner", action="store_true", help="Plot corner plots")
args = parser.parse_args()


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import arviz as az

az.style.use("arviz-darkgrid")

def main():

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
    d_nu = I_h["CDELT3"]
    nu_init = I_h["CRVAL3"]
    x_axis = (nu_init + d_nu * np.arange(len(I))) / 1e9
    name = I_h["OBJECT"]
    # ref_vel = I_h['ALTRVAL']
    # ref_pix = I_h['ALTRPIX']
    # vel_axis = (ref_vel + (np.arange(len(spec)) - ref_pix + 2) * (FreqtoLSR(nu_init) - FreqtoLSR(nu_init + d_nu))) / 1e3

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
    noise_V = np.std(
        np.concatenate(
            (V[: int(guess[1] - 6 * guess[2])], V[int(guess[-2] + 6 * guess[-1]) :])
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
    amp = mean.amp.values
    mu = mean.mu.values
    sig = mean.sigma.values
    model = np.zeros(len(I))
    x = np.arange(len(I))
    for i in range(num):
        model += amp[i] * np.exp(-((x - mu[i]) ** 2) / (2 * sig[i] ** 2))

    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axs[0].set(title=name + " Stokes I Fit", ylabel="Flux Density (Jy)")
    axs[0].plot(x_axis, I, "o", color="green", label="Data", markersize=4)
    axs[0].plot(x_axis, model, label="Fit", color="c")
    axs[0].plot(x_axis, I - model, "x", markersize=3, color="red", label="Residuals")
    axs[0].errorbar(
        x_axis,
        I - model,
        yerr=noise_I,
        fmt="none",
        ecolor="k",
        elinewidth=1,
        capsize=2,
        alpha=0.2,
    )

    for i in range(num):
        axs[0].plot(
            x_axis,
            amp[i] * np.exp(-((x - mu[i]) ** 2) / (2 * sig[i] ** 2)),
            label=f"Gauss{i}",
            alpha=0.5,
        )
        print(
            "Amp:",
            f"{amp[i]:.2f}",
            "Center:",
            f"{mu[i]:.2f}",
            "Width:",
            f"{sig[i]:.2f}",
            sep="\t",
            file=out_file,
        )  # Print to file
        print(
            "Amp:",
            f"{amp[i]:.2f}",
            "Center:",
            f"{mu[i]:.2f}",
            "Width:",
            f"{sig[i]:.2f}",
            sep="\t",
        )

    axs[0].legend()
    axs[1].plot(x_axis, (I - model), label="Residuals")
    axs[1].set(xlabel="Frequency (GHz)")
    axs[1].legend(title=f"$\chi^2$ = {np.sum((I - model)**2):.2f}")
    print("Chi2:", np.sum((I - model) ** 2), file=out_file)
    print("Chi2:", np.sum((I - model) ** 2))
    fig.savefig(cwd + "I_fit.png")
    I_fit = model
    
    xs = np.arange(len(I))
    compoments = np.array(
        [amp[i] * np.exp(-0.5 * ((xs - mu[i]) / sig[i]) ** 2) for i in range(num)]
    )
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
    V_model = alpha * I_fit + np.sum(
        [beta[i] * np.gradient(compoments, d_nu, axis=1)[i] for i in range(num)], axis=0
    )

    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axs[0].set(title=name + " Stokes V Fit", ylabel="Flux Density (Jy)")
    axs[0].plot(x_axis, V - alpha * I_fit, "o", color="green", label="Data", markersize=4)
    axs[0].plot(x_axis, V_model - alpha * I_fit, label="Fit", color="r", linewidth=2)
    # axs[0].plot(x_axis, V - V_model, 'x', markersize = 3, color = 'red', label = 'Residuals')
    axs[0].errorbar(
        x_axis,
        V - alpha * I_fit,
        yerr=noise_V,
        fmt="none",
        ecolor="k",
        elinewidth=1,
        capsize=2,
        alpha=0.2,
    )

    print("Alpha:", f"{alpha[0]:.2f}")
    print("Alpha:", f"{alpha[0]:.2f}", sep="\t", file=out_file)
    for i in range(num):
        axs[0].plot(
            x_axis, beta[i] * np.gradient(compoments[i], d_nu), label=f"Beta{i}", alpha=0.5
        )
        print(f"Beta{i}:", f"{beta[i]:.2f}", sep="\t", file=out_file)  # Print to file
        print(f"Beta{i}:", f"{beta[i]:.2f}", sep="\t")

    axs[0].legend()
    axs[1].plot(x_axis, (V - V_model), label="Residuals")
    axs[1].set(xlabel="Frequency (GHz)")
    axs[1].legend(title=f"$\chi^2$ = {np.sum((V - V_model)**2):.2f}")
    print("Chi2:", np.sum((V - V_model) ** 2))
    print("Chi2:", np.sum((V - V_model) ** 2), file=out_file)
    fig.savefig(cwd + "V_fit.png")

    out_file.close()

    # Plot 4 panel figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=False)
    axs[0][0].set(title=name + "  Stokes I Fit", ylabel="Flux Density (Jy)")
    axs[0][0].plot(x_axis, I, "o", color="green", label="Data", markersize=4)
    axs[0][0].plot(x_axis, I_fit, label="Fit", color="r", linewidth=2)
    axs[0][0].plot(x_axis, I - I_fit, "x", markersize=3, color="black", label="Residuals")
    axs[0][0].errorbar(
        x_axis, I, yerr=noise_I, fmt="none", ecolor="k", elinewidth=1, capsize=2, alpha=0.2
    )

    for i in range(num):
        axs[0][0].plot(
            x_axis,
            amp[i] * np.exp(-((x - mu[i]) ** 2) / (2 * sig[i] ** 2)),
            label=f"Gauss{i}",
            alpha=0.5,
        )

    axs[0][0].legend()
    axs[1][0].plot(x_axis, (I - I_fit), label="Residuals")
    axs[1][0].set(xlabel="Frequency (GHz)")
    axs[1][0].legend(title=f"$\chi^2$ = {np.sum((I - I_fit)**2):.2f}")

    axs[0][1].set(title=name + " Stokes V Fit")
    axs[0][1].plot(
        x_axis, V - alpha * I_fit, "o", color="green", label="Data", markersize=4
    )
    axs[0][1].plot(x_axis, V_model - alpha * I_fit, label="Fit", color="r", linewidth=2)
    axs[0][1].errorbar(
        x_axis,
        V - alpha * I_fit,
        yerr=noise_V,
        fmt="none",
        ecolor="k",
        elinewidth=1,
        capsize=2,
        alpha=0.2,
    )

    for i in range(num):
        axs[0][1].plot(
            x_axis, beta[i] * np.gradient(compoments[i], d_nu), label=f"Beta{i}", alpha=0.5
        )

    axs[0][1].legend()
    axs[1][1].plot(x_axis, (V - V_model), label="Residuals")
    axs[1][1].set(xlabel="Frequency (GHz)")
    axs[1][1].legend(title=f"$\chi^2$ = {np.sum((V - V_model)**2):.2f}")

    fig.savefig(cwd + "4_panel.png")
