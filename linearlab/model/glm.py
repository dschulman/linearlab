from collections.abc import Sequence
from dataclasses import dataclass
import formulaic
import numpy as np
import numpy.typing as npt
import pandas as pd

from linearlab import optim
from linearlab.lik import Likelihood
from linearlab.model.reg import Regularization, NullReg, Ridge

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
    
    def _fit_ridge(
        self,
        pen: float | Sequence[float],
        pen_intercept: bool = False,
        tol: float = 1e-8,
        max_iter: int = 1000
    ) -> tuple[float, dict[str, pd.DataFrame]]:
        d = self.lik.nparam
        y, logZ = self.lik.prepare_y(self.y)
        X = [Xi.to_numpy() for Xi in self.X]
        p = np.array([Xi.shape[1] for Xi in X])
        beta_splits = np.cumsum(p)[:-1]
        pens = list(pen) if isinstance(pen, Sequence) else [pen]*len(self.X)
        Xpens = zip(self.X, pens)
        if pen_intercept:
            lmbds = [np.full(Xi.shape[1], pi) for Xi, pi in Xpens]
        else:
            lmbds = [np.where(Xi.columns=="Intercept", 0.0, pi) for Xi, pi in Xpens]
        lmbd = np.concatenate(lmbds)
        def _fgh(beta):
            betas = np.split(beta, beta_splits)
            eta = np.vstack([Xi @ b for Xi, b in zip(X, betas)])
            f, g, h = self.lik(y, eta)
            G = [Xi.T @ gi for Xi, gi in zip(X, g)]
            H = np.empty((d,d), dtype=np.object_)
            for i in range(d):
                for j in range(d):
                    # faster version of X[i].T @ diag(h[i,j]) @ X[j]
                    H[i,j] = X[i].T @ (h[i,j][:,np.newaxis] * X[j])
            return (
                f - (0.5 * np.sum(lmbd * beta * beta)), 
                np.concatenate(G) - (lmbd * beta), 
                np.block(H.tolist()) + np.diag(lmbd),
            )
        beta0 = np.zeros(p.sum())
        beta, f, _, h = optim.newton_maxlik(_fgh, beta0, tol=tol, max_iter=max_iter)
        stderr = np.sqrt(np.diag(np.linalg.inv(h)))
        coefs = {
            param: pd.DataFrame({"est": b, "se": se}, index = Xi.columns)
            for param, b, se, Xi in zip(
                self.lik.params(), 
                np.split(beta, beta_splits), 
                np.split(stderr, beta_splits), 
                self.X
            )
        }
        return f + logZ, coefs

    def fit(self, tol: float = 1e-8, max_iter: int = 1000) -> "GLMFit":
        loglik, coefs = self._fit_ridge(0.0, False, tol, max_iter)
        return GLMFit(self, NullReg(), loglik, coefs)

    def fit_ridge(
        self, 
        pen: float | Sequence[float],
        pen_intercept: bool = False,
        tol: float = 1e-8,
        max_iter: int = 1000
    ) -> "GLMFit":
        loglik, coefs = self._fit_ridge(pen, pen_intercept, tol, max_iter)
        return GLMFit(self, Ridge(pen, pen_intercept), loglik, coefs)

glm = GLM.from_formula

@dataclass
class GLMFit:
    model: GLM
    reg: Regularization
    loglik: float
    coefs: dict[str, pd.DataFrame]

    def coef_table(self) -> pd.DataFrame:
        return pd.concat(self.coefs)

    def _repr_html_(self) -> str:
        return (
            "<div>" +
            f"<p>GLM model with {self.model.lik}</p>" +
            f"<p>Fit by maximum likelihood{self.reg.fit_desc()}</p>" +
            f"<p>{self.reg.loglik_desc()}Log-likelihood: {self.loglik:.2f}</p>" +
            self.coef_table().to_html() +
            "</div>"
        )

    def __repr__(self) -> str:
        return (
            f"GLM model with {self.model.lik}\n" +
            f"Fit by maximum likelihood{self.reg.fit_desc()}\n" +
            f"{self.reg.loglik_desc()}Log-likelihood: {self.loglik:.2f}\n" +
            str(self.coef_table())
        )
