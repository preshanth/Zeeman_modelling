class Zeeman:
    def __init__(
        self,
        filename: str,
        pixel: list,
        mapping: list,
        output: str,
        justI: bool,
        vel: bool,
        init: bool,
        trace: bool,
        corner: bool,
        absorb: bool,
        sensitivity: float,
        amps: list = None,
        means: list = None,
        sigmas: list = None,
        alpha: list = None,
        betas: list = None,
    ):
        self.filename = filename
        self.pixel = pixel
        self.mapping = mapping
        self.output = output
        self.justI = justI
        self.vel = vel
        self.init = init
        self.trace = trace
        self.corner = corner
        self.absorb = absorb
        self.sensitivity = sensitivity

        self.amps = (None,)
        self.means = (None,)
        self.sigmas = (None,)
        self.alpha = (None,)
        self.betas = (None,)

    def fit(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.io import fits
        import arviz as az

        az.style.use("arviz-darkgrid")

        # Read information from FITS files
        I_hdu = fits.open(self.filename[0])
        I_d = np.squeeze(I_hdu[0].data)
        I_h = I_hdu[0].header
        I_hdu.close()


        I = (
            I_d[:, self.pixel[0], self.pixel[1]]
            if not self.absorb
            else -1 * I_d[:, self.pixel[0], self.pixel[1]]
        )

        d_nu = I_h["CDELT3"]
        nu_init = I_h["CRVAL3"]
        nu_ref = I_h["CRPIX3"]
        x_axis = nu_init - nu_ref * d_nu + d_nu * np.arange(len(I))
        name = I_h["OBJECT"]

        units = 'Frequency (GHz)' if I_h["CTYPE3"] == "FREQ" else 'Velocity (km/s)'
        x_axis = x_axis / 1e9 if I_h["CTYPE3"] == "FREQ" else x_axis / 1e3

        from zeeman_modeling.Funcitons import mkdir

        cwd = mkdir(name, self.output)

        out_file = open(cwd + "output.txt", "w")

        from zeeman_modeling.Funcitons import guess_gen

        guess = guess_gen(I, self.mapping, sensitivity=self.sensitivity)
        num = len(guess) // 3
        noise_I = np.std(
            np.concatenate(
                (I[: int(guess[1] - 5 * guess[2])], I[int(guess[-2] + 5 * guess[-1]) :])
            ),
            ddof=1,
        )

        if self.init:
            plt.plot(I)
            for i in range(num):
                plt.axvline(guess[3 * i + 1], color="r")
            plt.savefig(cwd + "init_guess.png")

        from zeeman_modeling.Bayesian import fit_I

        Itrace = fit_I(guess, I, noise_I)

        if self.trace:
            axes = az.plot_trace(Itrace, compact=False)
            fig = axes.ravel()[0].figure
            fig.savefig(cwd + "I_trace.png")

        if self.corner:
            import corner

            corner.corner(
                Itrace, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4e"
            ).savefig(cwd + "I_corner.png")

        post = Itrace.posterior
        mean = post.mean(dim=["chain", "draw"])

        from zeeman_modeling.Plot import plotI

        amp = mean.amp.values
        mu = mean.mu.values
        sig = mean.sigma.values

        self.amps = amp
        self.means = mu
        self.sigmas = sig

        plotI(name, x_axis, units, I, amp, mu, sig, noise_I, out_file, cwd)

        if self.justI:
            out_file.close()
            return

        V_hdu = fits.open(self.filename[1])
        V_d = np.squeeze(V_hdu[0].data)
        V_h = V_hdu[0].header
        V_hdu.close()
        V = (
            V_d[:, self.pixel[0], self.pixel[1]]
            if not self.absorb
            else -1 * V_d[:, self.pixel[0], self.pixel[1]]
        )

        noise_V = np.std(
            np.concatenate(
                (V[: int(guess[1] - 5 * guess[2])], V[int(guess[-2] + 5 * guess[-1]) :])
            ),
            ddof=1,
        )

        xs = np.arange(len(I))
        components = np.array(
            [amp[i] * np.exp(-0.5 * ((xs - mu[i]) / sig[i]) ** 2) for i in range(num)]
        )
        I_fit = np.sum(components, axis=0)
        from zeeman_modeling.Bayesian import fit_V

        V_trace = fit_V(I_fit, V, d_nu, amp, mu, sig, noise_V)

        if self.trace:
            axes = az.plot_trace(V_trace, compact=False)
            fig = axes.ravel()[0].figure
            fig.savefig(cwd + "V_trace.png")

        if self.corner:
            import corner

            corner.corner(
                V_trace, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4e"
            ).savefig(cwd + "V_corner.png")

        post = V_trace.posterior
        mean = post.mean(dim=["chain", "draw"])
        alpha = mean.alpha.values
        beta = mean.beta.values

        self.alpha = alpha
        self.betas = beta

        from zeeman_modeling.Plot import plotV

        plotV(name, x_axis, units, V, components, d_nu, alpha, beta, noise_V, out_file, cwd)

        out_file.close()
        from zeeman_modeling.Plot import plot4pan

        plot4pan(
            name,
            x_axis,
            units,
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
