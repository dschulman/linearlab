from dataclasses import dataclass
import formulaic
import numpy as np
import numpy.linalg as npl
import pandas as pd

from linearlab.util import LOG_2PI

class LM:
    def __init__(self, y: pd.Series, X: pd.DataFrame) -> None:
        self.y = y
        self.X = X

    @staticmethod
    def from_formula(data: pd.DataFrame, form: str) -> "LM":
        y, X = formulaic.model_matrix(form, data)
        return LM(y.squeeze(), X)

    def fit(self) -> "LMFit":
        n = self.y.size
        y = self.y.to_numpy(np.float_)
        X = self.X.to_numpy(np.float_)
        beta = npl.lstsq(X.T @ X, X.T @ y, rcond=None)[0]
        y_hat = X @ beta
        rss = np.sum((y - y_hat)**2)
        sigma2 = rss / n
        loglik = -0.5 * n * (LOG_2PI + np.log(sigma2) + 1)
        return LMFit(
            model = self,
            loglik = loglik,
            beta = pd.Series(beta, index=self.X.columns),
            sigma2 = sigma2,
            y_hat = pd.Series(y_hat, index=self.y.index)
        )

lm = LM.from_formula

@dataclass
class LMFit:
    model: LM
    loglik: float
    beta: pd.Series
    sigma2: float
    y_hat: pd.Series
