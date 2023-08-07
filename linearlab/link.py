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

class IdentityLink(Link):
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return eta, np.ones_like(eta)

identity = IdentityLink()

class LogLink(Link):
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        y = np.exp(eta)
        return y, y

log = LogLink()

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

class CauchitLink(Link):
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        y = stats.cauchy.cdf(eta)
        dy = stats.cauchy.pdf(eta)
        return y, dy

cauchit = CauchitLink()

class LogLogLink(Link):
    def inv(
        self,
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        z = np.exp(-eta)
        y = np.exp(-z)
        dy = z * y
        return y, dy

loglog = LogLogLink()

class CLogLogLink(Link):
    def inv(
        self,
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        z = np.exp(eta)
        y = -special.expm1(-z)
        dy = z * (1 - y)
        return y, dy

cloglog = CLogLogLink()
