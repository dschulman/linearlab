from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class Likelihood(ABC):
    @property
    @abstractmethod
    def nparam(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __call__(
        self, 
        y: npt.NDArray,
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError()
 