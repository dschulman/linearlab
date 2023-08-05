from collections.abc import Sequence
from dataclasses import dataclass
import formulaic
import numpy as np
import numpy.typing as npt
import pandas as pd

from linearlab import optim
from linearlab.likelihood.base import Likelihood

class GLM:
    def __init__(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | Sequence[pd.DataFrame],
        lik: Likelihood,
    ) -> None:
        self.y = y
        self.X = list(X) if isinstance(X, Sequence) else [X]
        self.lik = lik
        while len(self.X) < lik.nparam:
            self.X.append(pd.DataFrame({"Intercept": np.ones(y.shape[0])}))

    @staticmethod
    def from_formula(data: pd.DataFrame, form: str, lik: Likelihood) -> "GLM":
        y, X = formulaic.model_matrix(form, data)
        return GLM(y.squeeze(), X, lik)
    
    def fit(self, tol: float = 1e-8, max_iter: int = 1000) -> "GLMFit":
        n = self.y.shape[0]
        d = self.lik.nparam
        y = self.y.to_numpy()
        X = [Xi.to_numpy() for Xi in self.X]
        p = np.array([Xi.shape[1] for Xi in X])
        beta_splits = np.cumsum(p)[:-1]
        def _fgh(beta):
            betas = np.split(beta, beta_splits)
            eta = np.column_stack([Xi @ b for Xi, b in zip(X, betas)])
            f, g, h = self.lik(y, eta)
            G = [Xi.T @ gi for Xi, gi in zip(X, g.T)]
            H = np.empty((d,d), dtype=np.object_)
            for i in range(d):
                for j in range(d):
                    # faster version of X[i].T @ diag(h[:,i,j]) @ X[j]
                    H[i,j] = X[i].T @ (h[:,i,j][:,np.newaxis] * X[j])
            return f, np.concatenate(G), np.block(H.tolist())
        beta0 = np.zeros(p.sum())
        loglik, beta = optim.newton_maxlik(_fgh, beta0, tol=tol, max_iter=max_iter) 
        betas = [pd.Series(b, index=Xi.columns) for b, Xi in zip(np.split(beta, beta_splits), self.X)]
        return GLMFit(self, loglik, betas)

glm = GLM.from_formula

@dataclass
class GLMFit:
    model: GLM
    loglik: float
    beta: list[pd.Series]
