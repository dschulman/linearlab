from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Any

class Likelihood(ABC):
    @abstractmethod
    def params(self) -> list[str]:
        raise NotImplementedError()

    @property
    def nparam(self) -> int:
        return len(self.params())

    @abstractmethod
    def prepare_y(self, y: pd.DataFrame | pd.Series) -> tuple[Any, float]:
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self, 
        y: Any,
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError()
 