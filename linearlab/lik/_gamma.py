import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special
from typing import Any

from linearlab.lik.base import Likelihood
from linearlab.link import Link, LogLink, log

class _GammaBase(Likelihood):
    def params(self) -> list[str]:
        return ["mu", "phi"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[Any, float]:
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
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        mu, dmu = self.mu_link.inv(eta[0])
        phi, dphi = self.phi_link.inv(eta[1])
        k = 1 / phi
        f = np.sum(
            - special.gammaln(k)
            - special.xlogy(k, mu * phi)
            + special.xlogy(k-1, y)
            - (y / mu / phi)
        )
        gmu = dmu * (y - mu) / mu**2 / phi
        gphi = dphi * (special.digamma(k) + np.log(mu * phi / y) + ((y - mu) / mu)) / phi**2
        g = np.stack([gmu, gphi])
        hmu = dmu**2 / mu**2 / phi
        hphi = dphi**2 * ((special.polygamma(1, k) / phi**4) - phi**(-3))
        hmu_phi = np.zeros_like(hmu)
        h = np.stack([[hmu, hmu_phi], [hmu_phi, hphi]])
        return f, g, h

class GammaLog(_GammaBase):
    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        mu = np.exp(eta[0])
        phi = np.exp(eta[1])
        k = 1 / phi
        f = np.sum(
            - special.gammaln(k)
            - special.xlogy(k, mu * phi)
            + special.xlogy(k-1, y)
            - (y / mu / phi)
        )
        gmu = (y - mu) / mu / phi
        gphi = (special.digamma(k) + np.log(mu * phi / y) + ((y - mu) / mu)) / phi
        g = np.stack([gmu, gphi])
        hmu = k
        hphi = (special.polygamma(1, k) / phi**2) - k
        hmu_phi = np.zeros_like(hmu)
        h = np.stack([[hmu, hmu_phi], [hmu_phi, hphi]])
        return f, g, h

def gamma(mu_link: Link = log, phi_link: Link = log) -> Likelihood:
    if isinstance(mu_link, LogLink) and isinstance(phi_link, LogLink):
        return GammaLog()
    else:
        return Gamma(mu_link, phi_link)

class _GammaSSBase(Likelihood):
    def params(self) -> list[str]:
        return ["k", "theta"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[Any, float]:
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
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        k, dk = self.k_link.inv(eta[0])
        theta, dtheta = self.theta_link.inv(eta[1])
        f = np.sum(
            - special.gammaln(k)
            - special.xlogy(k, theta)
            + special.xlogy(k - 1, y)
            - (y / theta)
        )
        gk = dk * (-special.digamma(k) - np.log(theta) + np.log(y))
        gtheta = dtheta * (y - (k * theta)) / theta**2
        g = np.stack([gk, gtheta])
        hk = dk**2 * special.polygamma(1, k)
        htheta = dtheta**2 * k / theta**2
        hk_theta = dk * dtheta / theta
        h = np.stack([[hk, hk_theta], [hk_theta, htheta]])
        return f, g, h

class GammaSSLogScale(_GammaSSBase):
    def __init__(self, k_link: Link) -> None:
        self.k_link = k_link

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        k, dk = self.k_link.inv(eta[0])
        log_theta = eta[1]
        theta = np.exp(eta[1])
        f = np.sum(
            - special.gammaln(k)
            - (k * log_theta)
            + special.xlogy(k - 1, y)
            - (y / theta)
        )
        gk = dk * (-special.digamma(k) - log_theta + np.log(y))
        gtheta = (y - (k * theta)) / theta
        g = np.stack([gk, gtheta])
        hk = dk**2 * special.polygamma(1, k)
        htheta = k
        hk_theta = dk
        h = np.stack([[hk, hk_theta], [hk_theta, htheta]])
        return f, g, h

def gamma_ss(k_link: Link = log, theta_link: Link = log) -> Likelihood:
    if isinstance(theta_link, LogLink):
        return GammaSSLogScale(k_link)
    else:
        return GammaSS(k_link, theta_link)
