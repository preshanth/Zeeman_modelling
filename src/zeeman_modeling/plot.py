import matplotlib.pyplot as plt
import numpy as np
import xarray


def plotI(
    name: str,
    x_axis: np.ndarray,
    I: np.ndarray,
    amp: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
    noise_I: float,
    out_file,
    cwd: str,
):
    """Plots Stokes I fitting results.

    Parameters
    ----------
        name: str
            Name of the observed object, taken from the FITS file.
        x_axis: np.ndarray
            The x_axis to be plotted, taken from the FITS file.
        I: np.ndarray
            Observed Stokes I spectrum.
        amp: list
            Amplitude of the Gaussian components from fitting Stokes I.
        mu: list
            Mean of the Gaussian components from fitting Stokes I.
        sig: list
            Standard deviation of the Gaussian components from fitting Stokes I.
        noise_I: float
            Noise level of the Stokes I spectrum.
        out_file: file object
            Output file object to write fit information.
        cwd: str
            Current working directory to save figures
    Returns
    -------
        None
    """
    model = np.zeros(len(x_axis))
    x = np.arange(len(x_axis))
    num = len(amp)
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
    axs[1].set(xlabel="LSR Velocity (m/s)")
    axs[1].legend(title=f"$\chi^2$ = {np.sum((I - model)**2):.2f}")
    print("Chi2:", np.sum((I - model) ** 2), file=out_file)
    print("Chi2:", np.sum((I - model) ** 2))
    fig.savefig(cwd + "I_fit.png")


def plotV(
    name: str,
    x_axis: np.ndarray,
    V: np.ndarray,
    components: np.ndarray,
    d_nu: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    noise_V: float,
    out_file,
    cwd: str,
):
    """Plots Stokes V fitting results.

    Parameters
    ----------
        name: str
            Name of the observed object, taken from the FITS file.
        x_axis: np.ndarray
            The x_axis to be plotted, taken from the FITS file.
        V: np.ndarray
            Observed Stokes V spectrum.
        components: np.ndarray
            Components of the Stokes I spectrum. Size is (num, len(x_axis)), where num is the number of Gaussian components.
        d_nu: float
            The channel width of the spectrum.
        alpha: np.ndarray
            Alpha parameter (Stokes I leakage term) from fitting Stokes V.
        beta: np.ndarray
            Beta parameter (Zeeman splitting term) from fitting Stokes V.
        noise_V: float
            Noise level of the Stokes V spectrum.
        out_file: file object
            Output file object to write fit information.
        cwd: str
            Current working directory to save figures
    Returns
    -------
        None
    """
    num = len(beta)
    I_fit = np.sum(components, axis=0)
    V_model = alpha * I_fit + np.sum(
        [beta[i] * np.gradient(components, d_nu, axis=1)[i] for i in range(num)], axis=0
    )
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axs[0].set(title=name + " Stokes V Fit", ylabel="Flux Density (Jy)")
    axs[0].plot(
        x_axis, V - alpha * I_fit, "o", color="green", label="Data", markersize=4
    )
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
            x_axis,
            beta[i] * np.gradient(components[i], d_nu),
            label=f"Beta{i}",
            alpha=0.5,
        )
        print(f"Beta{i}:", f"{beta[i]:.2f}", sep="\t", file=out_file)  # Print to file
        print(f"Beta{i}:", f"{beta[i]:.2f}", sep="\t")

    axs[0].legend()
    axs[1].plot(x_axis, (V - V_model), label="Residuals")
    axs[1].set(xlabel="LSR Velocity (m/s)")
    axs[1].legend(title=f"$\chi^2$ = {np.sum((V - V_model)**2):.2f}")
    print("Chi2:", np.sum((V - V_model) ** 2))
    print("Chi2:", np.sum((V - V_model) ** 2), file=out_file)
    fig.savefig(cwd + "V_fit.png")


def plot4pan(
    name: str,
    x_axis: np.ndarray,
    I: np.ndarray,
    V: np.ndarray,
    components: np.ndarray,
    d_nu: float,
    amp: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    noise_I: float,
    noise_V: float,
    cwd: str,
):
    """Plots Stokes I AND Stokes V fitting results.

    Parameters
    ----------
        name: str
            Name of the observed object, taken from the FITS file.
        x_axis: np.ndarray
            The x_axis to be plotted, taken from the FITS file.
        I: np.ndarray
            Observed Stokes I spectrum.
        V: np.ndarray
            Observed Stokes V spectrum.
        components: np.ndarray
            Components of the Stokes I spectrum. Size is (num, len(x_axis)), where num is the number of Gaussian components.
        d_nu: float
            The channel width of the spectrum.
        amp: list
            Amplitude of the Gaussian components from fitting Stokes I.
        mu: list
            Mean of the Gaussian components from fitting Stokes I.
        sig: list
            Standard deviation of the Gaussian components from fitting Stokes I.
        alpha: np.ndarray
            Alpha parameter (Stokes I leakage term) from fitting Stokes V.
        beta: np.ndarray
            Beta parameter (Zeeman splitting term) from fitting Stokes V.
        noise_I: float
            Noise level of the Stokes I spectrum.
        noise_V: float
            Noise level of the Stokes V spectrum.
        cwd: str
            Current working directory to save figures
    Returns
    -------
        None
    """

    I_fit = np.sum(components, axis=0)
    num = len(components)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=False)
    axs[0][0].set(title=name + "  Stokes I Fit", ylabel="Flux Density (Jy)")
    axs[0][0].plot(x_axis, I, "o", color="green", label="Data", markersize=4)
    axs[0][0].plot(x_axis, I_fit, label="Fit", color="r", linewidth=2)
    axs[0][0].plot(
        x_axis, I - I_fit, "x", markersize=3, color="black", label="Residuals"
    )
    axs[0][0].errorbar(
        x_axis,
        I,
        yerr=noise_I,
        fmt="none",
        ecolor="k",
        elinewidth=1,
        capsize=2,
        alpha=0.2,
    )
    x = np.arange(len(I))
    V_model = alpha * I_fit + np.sum(
        [beta[i] * np.gradient(components, d_nu, axis=1)[i] for i in range(num)], axis=0
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
    axs[1][0].set(xlabel="LSR Velocity (m/s)")
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
            x_axis,
            beta[i] * np.gradient(components[i], d_nu),
            label=f"Beta{i}",
            alpha=0.5,
        )

    axs[0][1].legend()
    axs[1][1].plot(x_axis, (V - V_model), label="Residuals")
    axs[1][1].set(xlabel="LSR Velocity (m/s)")
    axs[1][1].legend(title=f"$\chi^2$ = {np.sum((V - V_model)**2):.2f}")

    fig.savefig(cwd + "4_panel.png")
