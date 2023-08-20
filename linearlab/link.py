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

    def __repr__(self) -> str:
        return "log link"

log = LogLink()

class SoftPlusInvLink(Link):
    def inv(
        self,
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        y = np.logaddexp(0, eta)
        dy = special.expit(eta)
        return y, dy

    def __repr__(self) -> str:
        return "softplus inverse link"

softplusinv = SoftPlusInvLink()

class LogitLink(Link):
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        y = special.expit(eta)
        dy = y * (1 - y)
        return y, dy

    def __repr__(self) -> str:
        return "logit link"

logit = LogitLink()

class ProbitLink(Link):
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        y = stats.norm.cdf(eta)
        dy = stats.norm.pdf(eta)
        return y, dy

    def __repr__(self) -> str:
        return "probit link"

probit = ProbitLink()

class CauchitLink(Link):
    def inv(
        self, 
        eta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        y = stats.cauchy.cdf(eta)
        dy = stats.cauchy.pdf(eta)
        return y, dy

    def __repr__(self) -> str:
        return "cauchit link"

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

    def __repr__(self) -> str:
        return "log-log link"

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

    def __repr__(self) -> str:
        return "complementary log-log link"

cloglog = CLogLogLink()
