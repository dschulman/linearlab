from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Any, Generic, TypeVar

Y = TypeVar("Y")

class Likelihood(ABC, Generic[Y]):
    @abstractmethod
    def params(self) -> list[str]:
        raise NotImplementedError()

    @property
    def nparam(self) -> int:
        return len(self.params())

    @abstractmethod
    def prepare_y(self, y: pd.DataFrame | pd.Series) -> tuple[Y, float]:
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self, 
        y: Y,
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        raise NotImplementedError()

    def fg(self, y: Y, eta: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
        g = np.zeros_like(eta)
        f = self(y, eta, g, None)
        return f, g

    def fgh(self, y: Y, eta: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        g = np.zeros(eta.shape)
        h = np.zeros((eta.shape[0],) + eta.shape)
        f = self(y, eta, g, h)
        return f, g, h
