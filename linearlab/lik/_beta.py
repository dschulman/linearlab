import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special
from typing import Any

from linearlab.lik.base import Likelihood
from linearlab.link import Link, log, logit

class Beta(Likelihood[npt.NDArray[np.float64]]):
    def __init__(self, mu_link: Link, phi_link: Link) -> None:
        self.mu_link = mu_link
        self.phi_link = phi_link

    def params(self) -> list[str]:
        return ["mu", "phi"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[npt.NDArray[np.float64], float]:
        if not isinstance(y, pd.Series):
            raise ValueError("Beta likelihood requires univariate y")
        y = y.to_numpy(dtype = np.float_)
        logZ = 0.0
        return y, logZ

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        mu, dmu = self.mu_link.inv(eta[0])
        phi, dphi = self.phi_link.inv(eta[1])
        alpha = mu / phi
        beta = (1 - mu) / phi
        log_y = np.log(y)
        log_1my = np.log1p(-y)
        logit_y = log_y - log_1my
        f = np.sum(
            - special.betaln(alpha, beta)
            + ((alpha - 1) * log_y)
            + ((beta - 1) * log_1my)
        )
        if out_g is not None:
            dg_a = special.digamma(alpha)
            dg_b = special.digamma(beta)
            dg_invphi = special.digamma(1/phi)
            out_g[0] = dmu * (-dg_a + dg_b + logit_y) / phi
            out_g[1] = dphi * (
                + (mu * dg_a) + ((1 - mu) * dg_b) - dg_invphi
                - (mu * logit_y) - log_1my
            ) / phi**2
        if out_h is not None:
            tg_a = special.polygamma(1, alpha)
            tg_b = special.polygamma(1, beta)
            tg_invphi = special.polygamma(1, 1/phi)
            out_h[0,0] = dmu**2 * (tg_a + tg_b) / phi**2
            out_h[1,1] = dphi**2 * ((mu**2 * tg_a) + ((1-mu)**2 * tg_b) - tg_invphi) / phi**4
            out_h[0,1] = out_h[1,0] = dmu * dphi * (- (mu * tg_a) + ((1-mu) * tg_b)) / phi**3
        return f

    def __repr__(self) -> str:
        return f"beta likelihood with mean ({self.mu_link}) and dispersion ({self.phi_link})"

def beta(mu_link: Link = logit, phi_link: Link = log) -> Likelihood:
    return Beta(mu_link, phi_link)

class BetaCanon(Likelihood[npt.NDArray[np.float64]]):
    def __init__(self, alpha_link: Link, beta_link: Link) -> None:
        self.alpha_link = alpha_link
        self.beta_link = beta_link

    def params(self) -> list[str]:
        return ["alpha", "beta"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[npt.NDArray[np.float64], float]:
        if not isinstance(y, pd.Series):
            raise ValueError("Beta likelihood requires univariate y")
        y = y.to_numpy(dtype = np.float_)
        logZ = 0.0
        return y, logZ

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64],
        out_g: None | npt.NDArray[np.float64],
        out_h: None | npt.NDArray[np.float64],
    ) -> float:
        alpha, dalpha = self.alpha_link.inv(eta[0])
        beta, dbeta = self.beta_link.inv(eta[1])
        log_y = np.log(y)
        log_1my = np.log1p(-y)
        f = np.sum(
            - special.betaln(alpha, beta)
            + ((alpha - 1) * log_y)
            + ((beta - 1) * log_1my)
        )
        if out_g is not None:
            dg_a = special.digamma(alpha)
            dg_b = special.digamma(beta)
            dg_ab = special.digamma(alpha + beta)
            out_g[0] = dalpha * (dg_ab - dg_a + log_y)
            out_g[1] = dbeta * (dg_ab - dg_b + log_1my)
        if out_h is not None:
            tg_a = special.polygamma(1, alpha)
            tg_b = special.polygamma(1, beta)
            tg_ab = special.polygamma(1, alpha + beta)
            out_h[0,0] = dalpha**2 * (tg_a - tg_ab)
            out_h[1,1] = dbeta**2 * (tg_b - tg_ab)
            out_h[0,1] = out_h[1,0] = dalpha * dbeta * (- tg_ab)
        return f

    def __repr__(self) -> str:
        return f"beta likelihood with alpha ({self.alpha_link}) and beta ({self.beta_link})"

def beta_canon(alpha_link: Link = log, beta_link: Link = log) -> Likelihood:
    return BetaCanon(alpha_link, beta_link)
