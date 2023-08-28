from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

class Regularization(ABC):
    @abstractmethod
    def fit_desc(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def loglik_desc(self) -> str:
        raise NotImplementedError()

class NullReg(Regularization):
    def fit_desc(self) -> str:
        return ""

    def loglik_desc(self) -> str:
        return ""

@dataclass(frozen = True)
class Ridge(Regularization):
    pen: float | Sequence[float]
    pen_intercept: bool

    def fit_desc(self) -> str:
        pens = list(self.pen) if isinstance(self.pen, Sequence) else [self.pen]
        pstr = ", ".join("{:.2e}".format(pen) for pen in pens)
        istr = "; intercept penalized" if self.pen_intercept else ""
        return f" with ridge regularization (penalty={pstr}{istr})"

    def loglik_desc(self) -> str:
        return "Penalized "
