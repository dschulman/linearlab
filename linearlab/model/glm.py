from collections.abc import Sequence
from dataclasses import dataclass
import formulaic
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Any

from linearlab import optim
from linearlab.lik import Likelihood
from linearlab.model.reg import Regularization, NullReg, Ridge, Lasso

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
    
    def _penalty(self, pen: float | Sequence[float], pen_intercept: bool) -> npt.NDArray[np.float64]:
        pens = list(pen) if isinstance(pen, Sequence) else [pen]*len(self.X)
        Xpens = zip(self.X, pens)
        if pen_intercept:
            lmbds = [np.full(Xi.shape[1], pi) for Xi, pi in Xpens]
        else:
            lmbds = [np.where(Xi.columns=="Intercept", 0.0, pi) for Xi, pi in Xpens]
        return np.concatenate(lmbds)

    def _coef_table(
        self, 
        beta: npt.NDArray[np.float64], 
        h: npt.NDArray[np.float64]
    ) -> dict[str, pd.DataFrame]:
        p = np.array([Xi.shape[1] for Xi in self.X])
        beta_splits = np.cumsum(p)[:-1]
        stderr = np.sqrt(np.diag(np.linalg.inv(h)))
        return {
            param: pd.DataFrame({"est": b, "se": se}, index = Xi.columns)
            for param, b, se, Xi in zip(
                self.lik.params(), 
                np.split(beta, beta_splits), 
                np.split(stderr, beta_splits), 
                self.X
            )
        }

    def _fgh(
        self,
        beta: npt.NDArray[np.float64],
        X: Sequence[npt.NDArray[np.float64]],
        y: Any
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        d = self.lik.nparam
        p = np.array([Xi.shape[1] for Xi in X])
        beta_splits = np.cumsum(p)[:-1]
        betas = np.split(beta, beta_splits)
        eta = np.vstack([Xi @ b for Xi, b in zip(X, betas)])
        f, g, h = self.lik(y, eta)
        G = [Xi.T @ gi for Xi, gi in zip(X, g)]
        H = np.empty((d,d), dtype=np.object_)
        for i in range(d):
            for j in range(d):
                # faster version of X[i].T @ diag(h[i,j]) @ X[j]
                H[i,j] = X[i].T @ (h[i,j][:,np.newaxis] * X[j])
        return f, np.concatenate(G), np.block(H.tolist())

    def _fit_ridge(
        self,
        pen: float | Sequence[float],
        pen_intercept: bool = False,
        tol: float = 1e-8,
        max_iter: int = 1000
    ) -> tuple[float, dict[str, pd.DataFrame]]:   
        y, logZ = self.lik.prepare_y(self.y)
        X = [Xi.to_numpy() for Xi in self.X]
        lmbd = self._penalty(pen, pen_intercept)
        beta0 = np.zeros_like(lmbd)
        def _fgh(beta):
            f, g, h = self._fgh(beta, X, y)
            return (
                f - (0.5 * np.sum(lmbd * beta * beta)),
                g - (lmbd * beta),
                h + np.diag(lmbd)
            )
        beta, f, _, h = optim.newton_maxlik(_fgh, beta0, tol=tol, max_iter=max_iter)
        return f + logZ, self._coef_table(beta, h)

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

    def fit_lasso(
        self,
        pen: float | Sequence[float],
        pen_intercept: bool = False,
        rho: float = 1.0,
        tol_abs: float = 1e-4,
        tol_rel: float = 1e-2,
        max_iter: int = 1000,
        inner_tol: float = 1e-6,
        inner_max_iter: int = 1000,
    ) -> "GLMFit":
        y, logZ = self.lik.prepare_y(self.y)
        X = [Xi.to_numpy() for Xi in self.X]
        lmbd = self._penalty(pen, pen_intercept)
        def _inner_fit(beta, offset, rho):
            def _fgh(beta):
                f, g, h = self._fgh(beta, X, y)
                z = beta - offset
                return (
                    f - (0.5 * rho * np.dot(z, z)),
                    g - (rho * z),
                    h + (rho * np.eye(g.shape[0]))
                )
            return optim.newton_maxlik(_fgh, beta, tol=inner_tol, max_iter=inner_max_iter)[0]
        beta = optim.l1_admm(
            _inner_fit,
            n = lmbd.shape[0],
            l1_reg = lmbd,
            rho = rho,
            tol_abs = tol_abs,
            tol_rel = tol_rel,
            max_iter = max_iter
        )
        f, _, h = self._fgh(beta, X, y)
        return GLMFit(
            model = self,
            reg = Lasso(pen, pen_intercept),
            loglik = f + logZ - np.sum(lmbd * np.abs(beta)),
            coefs = self._coef_table(beta, h)
        )

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
