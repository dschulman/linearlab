import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Any

from linearlab.lik.base import Likelihood
from linearlab.link import Link, identity, LogLink, log
from linearlab.util import LOG_2PI

class _NormalBase(Likelihood[npt.NDArray[np.float64]]):
    def params(self) -> list[str]:
        return ["mu", "sigma"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[npt.NDArray[np.float64], float]:
        if not isinstance(y, pd.Series):
            raise ValueError("normal likelihood only supports univariate y")
        return y.to_numpy(dtype=np.float_), -0.5 * LOG_2PI * y.shape[0]

class Normal(_NormalBase):
    def __init__(self, mu_link: Link, sigma_link: Link) -> None:
        self.mu_link = mu_link
        self.sigma_link = sigma_link

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        mu, dmu = self.mu_link.inv(eta[0])
        sigma, dsigma = self.sigma_link.inv(eta[1])
        z = (y - mu) / sigma
        f = -0.5 * np.sum(2*np.log(sigma) + z**2)
        if out_g is not None:
            out_g[0] = dmu * z / sigma
            out_g[1] = dsigma * (z**2 - 1) / sigma
        if out_h is not None:
            out_h[0,0] = dmu**2 / sigma**2
            out_h[1,1] = 2 * dsigma**2 / sigma**2
        return f

    def __repr__(self) -> str:
        return f"normal likelihood with mean ({self.mu_link}) and scale ({self.sigma_link})"

class NormalLogScale(_NormalBase):
    def __init__(self, mu_link: Link) -> None:
        self.mu_link = mu_link

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        mu, dmu = self.mu_link.inv(eta[0])
        log_sigma = eta[1]
        sigma = np.exp(log_sigma)
        z = (y - mu) / sigma
        f = -0.5 * np.sum(2*log_sigma + z**2)
        if out_g is not None:
            out_g[0] = dmu * z / sigma
            out_g[1] = (z**2 - 1) / sigma
        if out_h is not None:
            out_h[0,0] = 1 / sigma**2
            out_h[1,1] = np.full_like(hmu, 2.0)
        return f

    def __repr__(self) -> str:
        return f"normal likelihood with mean ({self.mu_link}) and scale (log link)"

def normal(mu_link: Link = identity, sigma_link: Link = log) -> Likelihood:
    if isinstance(sigma_link, LogLink):
        return NormalLogScale(mu_link)
    else:
        return Normal(mu_link, sigma_link)
