import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special
from typing import Any

from linearlab.lik.base import Likelihood
from linearlab.link import Link, LogLink, log

class _PoissonBase(Likelihood):
    def params(self) -> list[str]:
        return ["mu"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[Any, float]:
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
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        mu, dmu = self.link.inv(eta[0])
        f = np.sum(special.xlogy(y, mu) - mu)
        g = dmu * (y - mu) / mu
        h = (dmu**2) / mu
        return f, g[np.newaxis,:], h[np.newaxis,np.newaxis,:]

class PoissonLog(_PoissonBase):
    def __call__(
        self, 
        y: npt.NDArray[np.int_],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        log_mu = eta[0]
        mu = np.exp(log_mu)
        f = np.sum((y * log_mu) - mu)
        g = y - mu
        h = mu
        return f, g[np.newaxis,:], h[np.newaxis,np.newaxis,:]

def poisson(link: Link = log) -> Likelihood:
    if isinstance(link, LogLink):
        return PoissonLog()
    else:
        return Poisson(link)
