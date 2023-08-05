from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from scipy import special, stats

class Link(ABC):
    @abstractmethod
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError()

class LogitLink(Link):
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        y = special.expit(eta)
        dy = y * (1 - y)
        return y, dy

logit = LogitLink()

class ProbitLink(Link):
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        y = stats.norm.cdf(eta)
        dy = stats.norm.pdf(eta)
        return y, dy

probit = ProbitLink()
