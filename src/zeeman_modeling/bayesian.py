import pymc as pm
import numpy as np

def fit_I(guess: list, I: np.ndarray, noise_I: float):
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
        )
    
    return trace

def fit_V(I_fit: np.ndarray, V: np.ndarray, d_nu: float, amp: list, mu: list, sig: list, noise_V: float):
    num = len(amp)
    stokesVmodel = pm.Model()
    xs = np.arange(len(I_fit))
    compoments = np.array(
        [amp[i] * np.exp(-0.5 * ((xs - mu[i]) / sig[i]) ** 2) for i in range(num)]
    )

    with stokesVmodel as model:
        Ifit = pm.Data("I", I_fit)
        d_I = pm.Data("d_I", np.gradient(compoments, d_nu, axis=1))
        alpha = pm.Flat("alpha", shape=1)
        beta = pm.Flat("beta", shape=num)
        V_fit = alpha * Ifit + pm.math.sum([beta[i] * d_I[i] for i in range(num)], axis=0)
        V_likelihood = pm.Normal("V", mu=V_fit, observed=V, sigma=noise_V)

        V_trace = pm.sample(
            draws=10_000,
            tune=5_000,
            cores=4,
            chains=4,
            discard_tuned_samples=True,
            step=pm.NUTS(),
        )