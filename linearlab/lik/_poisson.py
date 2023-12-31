import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special
from typing import Any

from linearlab.lik.base import Likelihood
from linearlab.link import Link, LogLink, log

class _PoissonBase(Likelihood[npt.NDArray[np.int_]]):
    def params(self) -> list[str]:
        return ["mu"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[npt.NDArray[np.int_], float]:
        if not isinstance(y, pd.Series):
            raise ValueError("Poisson likelihood requires univariate y")
        y = y.to_numpy(dtype=np.int_)
        logZ = -np.sum(special.gammaln(y+1))
        return y, logZ

class Poisson(_PoissonBase):
    def __init__(self, link: Link) -> None:
        self.link = link
    
    def __call__(
        self, 
        y: npt.NDArray[np.int_],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        mu, dmu = self.link.inv(eta[0])
        f = np.sum(special.xlogy(y, mu) - mu)
        if out_g is not None:
            out_g[0] = dmu * (y - mu) / mu
        if out_h is not None:
            out_h[0,0] = (dmu**2) / mu
        return f

    def __repr__(self) -> str:
        return f"poisson likelihood ({self.link})"

class PoissonLog(_PoissonBase):
    def __call__(
        self, 
        y: npt.NDArray[np.int_],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        log_mu = eta[0]
        mu = np.exp(log_mu)
        f = np.sum((y * log_mu) - mu)
        if out_g is not None:
            out_g[0] = y - mu
        if out_h is not None:
            out_h[0,0] = mu
        return f

    def __repr__(self) -> str:
        return "poisson likelihood (log link)"

def poisson(link: Link = log) -> Likelihood:
    if isinstance(link, LogLink):
        return PoissonLog()
    else:
        return Poisson(link)
