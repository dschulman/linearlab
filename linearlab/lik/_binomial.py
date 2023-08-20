import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special
from typing import Any

from linearlab.lik.base import Likelihood
from linearlab.link import Link, LogitLink, logit

class _BinomialBase(Likelihood):
    def params(self) -> list[str]:
        return ["p"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[Any, float]:
        if isinstance(y, pd.Series):
            k = y.to_numpy(dtype=np.bool_).astype(np.int_)
            n = np.ones(y.shape[0], dtype=np.int_)
            logZ = 0.0
        elif isinstance(y, pd.DataFrame) and (y.shape[1] == 2):
            y = y.to_numpy(dtype=np.int_)
            k = y[:,0]
            n = y.sum(axis=1)
            logZ = np.sum(special.gammaln(n+1) - special.gammaln(k+1) - special.gammaln(n-k+1))
        else:
            raise ValueError("binomial needs either bool or n_success,n_failure")
        return (k, n), logZ

class Binomial(_BinomialBase):
    def __init__(self, link:Link) -> None:
        self.link = link

    def __call__(
        self, 
        y: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        k, n = y
        p, dp = self.link.inv(eta[0])
        f = np.sum(
            (k * special.logit(p)) +
            (n * special.log1p(-p))
        )
        g = dp * (k - (n * p)) / p / (1 - p)
        h = (dp**2) * n / p / (1 - p)
        return f, g[np.newaxis,:], h[np.newaxis,np.newaxis,:]

    def __repr__(self) -> str:
        return f"binomial likelihood ({self.link})"

class BinomialLogit(_BinomialBase):
    def __call__(
        self, 
        y: tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        k, n = y
        eta = eta[0]
        p = special.expit(eta)
        f = np.sum(
            (k * eta) +
            (n * special.log_expit(-eta))
        )
        g = k - (n * p)
        h = n * p * (1 - p)
        return f, g[np.newaxis,:], h[np.newaxis,np.newaxis,:]

    def __repr__(self) -> str:
        return "binomial likelihood (logit link)"

def binomial(link: Link = logit) -> Likelihood:
    if isinstance(link, LogitLink):
        return BinomialLogit()
    else:
        return Binomial(link)
