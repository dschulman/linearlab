import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special
from typing import Any

from linearlab.lik.base import Likelihood
from linearlab.link import Link, LogLink, log

class _GammaBase(Likelihood[npt.NDArray[np.float64]]):
    def params(self) -> list[str]:
        return ["mu", "phi"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[npt.NDArray[np.float64], float]:
        if not isinstance(y, pd.Series):
            raise ValueError("Gamma likelihood requires univariate y")
        y = y.to_numpy(dtype = np.float_)
        logZ = 0.0
        return y, logZ

class Gamma(_GammaBase):
    def __init__(self, mu_link: Link, phi_link: Link) -> None:
        self.mu_link = mu_link
        self.phi_link = phi_link

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        mu, dmu = self.mu_link.inv(eta[0])
        phi, dphi = self.phi_link.inv(eta[1])
        k = 1 / phi
        f = np.sum(
            - special.gammaln(k)
            - special.xlogy(k, mu * phi)
            + special.xlogy(k-1, y)
            - (y / mu / phi)
        )
        if out_g is not None:
            out_g[0] = dmu * (y - mu) / mu**2 / phi
            out_g[1] = dphi * (special.digamma(k) + np.log(mu * phi / y) + ((y - mu) / mu)) / phi**2
        if out_h is not None:
            out_h[0,0] = dmu**2 / mu**2 / phi
            out_h[1,1] = dphi**2 * ((special.polygamma(1, k) / phi**4) - phi**(-3))
        return f

    def __repr__(self) -> str:
        return f"gamma likelihood with mean ({self.mu_link}) and dispersion ({self.phi_link})"

class GammaLog(_GammaBase):
    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        mu = np.exp(eta[0])
        phi = np.exp(eta[1])
        k = 1 / phi
        f = np.sum(
            - special.gammaln(k)
            - special.xlogy(k, mu * phi)
            + special.xlogy(k-1, y)
            - (y / mu / phi)
        )
        if out_g is not None:
            out_g[0] = (y - mu) / mu / phi
            out_g[1] = (special.digamma(k) + np.log(mu * phi / y) + ((y - mu) / mu)) / phi
        if out_h is not None:
            out_h[0,0] = k
            out_h[1,1] = (special.polygamma(1, k) / phi**2) - k
        return f

    def __repr__(self) -> str:
        return "gamma likelihood with mean (log link) and dispersion (log link)"

def gamma(mu_link: Link = log, phi_link: Link = log) -> Likelihood:
    if isinstance(mu_link, LogLink) and isinstance(phi_link, LogLink):
        return GammaLog()
    else:
        return Gamma(mu_link, phi_link)

class _GammaSSBase(Likelihood[npt.NDArray[np.float64]]):
    def params(self) -> list[str]:
        return ["k", "theta"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[npt.NDArray[np.float64], float]:
        if not isinstance(y, pd.Series):
            raise ValueError("Gamma likelihood requires univariate y")
        y = y.to_numpy(dtype = np.float_)
        logZ = 0.0
        return y, logZ

class GammaSS(_GammaSSBase):
    def __init__(self, k_link: Link, theta_link: Link) -> None:
        self.k_link = k_link
        self.theta_link = theta_link

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        k, dk = self.k_link.inv(eta[0])
        theta, dtheta = self.theta_link.inv(eta[1])
        f = np.sum(
            - special.gammaln(k)
            - special.xlogy(k, theta)
            + special.xlogy(k - 1, y)
            - (y / theta)
        )
        if out_g is not None:
            out_g[0] = dk * (-special.digamma(k) - np.log(theta) + np.log(y))
            out_g[1] = dtheta * (y - (k * theta)) / theta**2
        if out_h is not None:
            out_h[0,0] = dk**2 * special.polygamma(1, k)
            out_h[1,1] = dtheta**2 * k / theta**2
            out_h[0,1] = out_h[1,0] = dk * dtheta / theta
        return f

    def __repr__(self) -> str:
        return f"gamma likelihood with shape ({self.k_link}) and scale ({self.theta_link})"
    
class GammaSSLogScale(_GammaSSBase):
    def __init__(self, k_link: Link) -> None:
        self.k_link = k_link

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        k, dk = self.k_link.inv(eta[0])
        log_theta = eta[1]
        theta = np.exp(eta[1])
        f = np.sum(
            - special.gammaln(k)
            - (k * log_theta)
            + special.xlogy(k - 1, y)
            - (y / theta)
        )
        if out_g is not None:
            out_g[0] = dk * (-special.digamma(k) - log_theta + np.log(y))
            out_g[1] = (y - (k * theta)) / theta
        if out_h is not None:
            out_h[0,0] = dk**2 * special.polygamma(1, k)
            out_h[1,1] = k
            out_h[0,1] = out_h[1,0] = dk
        return f

    def __repr__(self) -> str:
        return f"gamma likelihood with shape ({self.k_link}) and scale (log link)"

def gamma_ss(k_link: Link = log, theta_link: Link = log) -> Likelihood:
    if isinstance(theta_link, LogLink):
        return GammaSSLogScale(k_link)
    else:
        return GammaSS(k_link, theta_link)
