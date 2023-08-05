import numpy as np
import numpy.typing as npt
from scipy import special

from linearlab.likelihood.base import Likelihood
from linearlab.link import Link, LogitLink, logit

class Bernoulli(Likelihood):
    def __init__(self, link:Link) -> None:
        self.link = link

    @property
    def nparam(self) -> int:
        return 1

    def __call__(
        self, 
        y: npt.NDArray[np.bool_],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        p, dp = self.link.inv(eta[:,0])
        f = np.sum((y * special.logit(p)) + special.log1p(-p))
        g = dp * (y - p) / p / (1 - p)
        h = (dp**2) / p / (1 - p)
        return f, g[:,np.newaxis], h[:,np.newaxis,np.newaxis]

class BernoulliLogit(Likelihood):
    @property
    def nparam(self) -> int:
        return 1

    def __call__(
        self, 
        y: npt.NDArray[np.bool_],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        eta = eta[:,0]
        p = special.expit(eta)
        f = np.sum((y * eta) + special.log_expit(-eta))
        g = y - p
        h = p * (1 - p)
        return f, g[:,np.newaxis], h[:,np.newaxis,np.newaxis]

def bernoulli(link: Link = logit) -> Likelihood:
    if isinstance(link, LogitLink):
        return BernoulliLogit()
    else:
        return Bernoulli(link)
