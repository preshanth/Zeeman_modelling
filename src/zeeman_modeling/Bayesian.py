import pymc as pm
import numpy as np


def fit_I(guess: list, I: np.ndarray, noise_I: float):
    """Fits Stokes I spectrum with a Gaussian mixture model using MCMC algorithms.

    Parameters
    ----------
        guess: list
            Initial guess of the Gaussian mixture model. In the form of [amp0, mu0, sigma0, amp1, mu1, sigma1, ...].
        I: np.ndarray
            The Stokes I spectrum to be fitted.
        noise_I: float
            Noise level of the Stokes I spectrum.

    Returns
    -------
        trace: pymc.backends.base.MultiTrace
            A MultiTrace object containing samples.
    """
    I_model = pm.Model()
    xs = np.arange(len(I))
    num = len(guess) // 3

    with I_model as model:
        xdata = pm.ConstantData("x", xs)
        # Priors for unknown model parameters
        amp = pm.Uniform("amp", lower=0, upper=np.max(I) * 1.2, shape=num)
        mu = pm.Normal(
            "mu",
            mu=guess[1::3],
            sigma=10,
            shape=num,
            transform=pm.distributions.transforms.univariate_ordered,
        )
        sigma = pm.HalfNormal("sigma", sigma=10, shape=num)

        gauss_sum = np.sum(
            [
                amp[i] * pm.math.exp(-0.5 * ((xdata - mu[i]) / sigma[i]) ** 2)
                for i in range(num)
            ],
            axis=0,
        )
        # Likelihood (sampling distribution) of observations
        likelihood = pm.Normal("y", mu=gauss_sum, observed=I, sigma=noise_I)

        # inference
        trace = pm.sample(
            draws=10_000,
            tune=5_000,
            cores=4,
            chains=4,
            discard_tuned_samples=True,
            step=pm.NUTS(),
            initvals={"amp": guess[0::3], "mu": guess[1::3], "sigma": guess[2::3]},
            progressbar=False,
        )

    return trace


def fit_V(
    I_fit: np.ndarray,
    V: np.ndarray,
    d_nu: float,
    amp: list,
    mu: list,
    sig: list,
    noise_V: float,
):
    """Fits Stokes V spectrum with given Stokes I using MCMC algorithms.

    Parameters
    ----------
        I_fit: np.ndarray
            Stokes I fit from the Gaussian mixture model.
        V: np.ndarray
            The Stokes V spectrum to be fitted.
        d_nu: float
            Difference in frequency across channels for calculating the gradient of Stokes I.
        amp: list
            Amplitude of the Gaussian components from fitting Stokes I.
        mu: list
            Mean of the Gaussian components from fitting Stokes I.
        sig: list
            Standard deviation of the Gaussian components from fitting Stokes I.
    Returns
    -------
        trace: pymc.backends.base.MultiTrace
            A MultiTrace object containing samples.
    """
    num = len(amp)
    stokesVmodel = pm.Model()
    xs = np.arange(len(I_fit))
    compoments = np.array(
        [amp[i] * np.exp(-0.5 * ((xs - mu[i]) / sig[i]) ** 2) for i in range(num)]
    )

    with stokesVmodel as model:
        Ifit = pm.Data("I", I_fit, mutable=False)
        d_I = pm.Data("d_I", np.gradient(compoments, d_nu, axis=1), mutable=False)
        alpha = pm.Flat("alpha", shape=1)
        beta = pm.Flat("beta", shape=num)
        V_fit = alpha * Ifit + pm.math.sum(
            [beta[i] * d_I[i] for i in range(num)], axis=0
        )
        V_likelihood = pm.Normal("V", mu=V_fit, observed=V, sigma=noise_V)

        V_trace = pm.sample(
            draws=10_000,
            tune=5_000,
            cores=4,
            chains=4,
            discard_tuned_samples=True,
            step=pm.NUTS(),
            progressbar=False,
        )
        return V_trace
