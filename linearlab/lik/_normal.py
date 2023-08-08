import numpy as np
import numpy.typing as npt

from linearlab.lik.base import Likelihood
from linearlab.link import Link, identity, LogLink, log
from linearlab.util import LOG_2PI

class Normal(Likelihood):
    def __init__(self, mu_link: Link, sigma_link: Link) -> None:
        self.mu_link = mu_link
        self.sigma_link = sigma_link

    def params(self) -> list[str]:
        return ["mu", "sigma"]

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        mu, dmu = self.mu_link.inv(eta[0])
        sigma, dsigma = self.sigma_link.inv(eta[1])
        z = (y - mu) / sigma
        f = -0.5 * np.sum(LOG_2PI + 2*np.log(sigma) + z**2)
        gmu = dmu * z / sigma
        gsigma = dsigma * (z**2 - 1) / sigma
        g = np.stack([gmu, gsigma])
        hmu = dmu**2 / sigma**2
        hsigma = 2 * dsigma**2 / sigma**2
        hmu_sigma = np.zeros(y.shape[0])
        h = np.stack([[hmu, hmu_sigma], [hmu_sigma, hsigma]])
        return f, g, h

class NormalLogScale(Likelihood):
    def __init__(self, mu_link: Link) -> None:
        self.mu_link = mu_link

    def params(self) -> list[str]:
        return ["mu", "sigma"]

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        mu, dmu = self.mu_link.inv(eta[0])
        log_sigma = eta[1]
        sigma = np.exp(log_sigma)
        z = (y - mu) / sigma
        f = -0.5 * np.sum(LOG_2PI + 2*log_sigma + z**2)
        gmu = dmu * z / sigma
        gsigma = (z**2 - 1) / sigma
        g = np.stack([gmu, gsigma])
        hmu = 1 / sigma**2
        hsigma = np.full_like(hmu, 2.0)
        hmu_sigma = np.zeros(y.shape[0])
        h = np.stack([[hmu, hmu_sigma], [hmu_sigma, hsigma]])
        return f, g, h  

def normal(mu_link: Link = identity, sigma_link: Link = log) -> Likelihood:
    if isinstance(sigma_link, LogLink):
        return NormalLogScale(mu_link)
    else:
        return Normal(mu_link, sigma_link)
