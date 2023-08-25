import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special
from typing import Any

from linearlab.lik.base import Likelihood
from linearlab.link import Link, log, logit

class Beta(Likelihood):
    def __init__(self, mu_link: Link, phi_link: Link) -> None:
        self.mu_link = mu_link
        self.phi_link = phi_link

    def params(self) -> list[str]:
        return ["mu", "phi"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[Any, float]:
        if not isinstance(y, pd.Series):
            raise ValueError("Beta likelihood requires univariate y")
        y = y.to_numpy(dtype = np.float_)
        logZ = 0.0
        return y, logZ

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        dg_a = special.digamma(alpha)
        dg_b = special.digamma(beta)
        dg_invphi = special.digamma(1/phi)
        gmu = dmu * (-dg_a + dg_b + logit_y) / phi
        gphi = dphi * (
            + (mu * dg_a) + ((1 - mu) * dg_b) - dg_invphi
            - (mu * logit_y) - log_1my
        ) / phi**2
        g = np.stack([gmu, gphi])
        tg_a = special.polygamma(1, alpha)
        tg_b = special.polygamma(1, beta)
        tg_invphi = special.polygamma(1, 1/phi)
        hmu = dmu**2 * (tg_a + tg_b) / phi**2
        hphi = dphi**2 * ((mu**2 * tg_a) + ((1-mu)**2 * tg_b) - tg_invphi) / phi**4
        hmu_phi = dmu * dphi * (- (mu * tg_a) + ((1-mu) * tg_b)) / phi**3
        h = np.stack([[hmu, hmu_phi], [hmu_phi, hphi]])
        return f, g, h

    def __repr__(self) -> str:
        return f"beta likelihood with mean ({self.mu_link}) and dispersion ({self.phi_link})"

def beta(mu_link: Link = logit, phi_link: Link = log) -> Likelihood:
    return Beta(mu_link, phi_link)

class BetaCanon(Likelihood):
    def __init__(self, alpha_link: Link, beta_link: Link) -> None:
        self.alpha_link = alpha_link
        self.beta_link = beta_link

    def params(self) -> list[str]:
        return ["alpha", "beta"]

    def prepare_y(self, y: pd.Series | pd.DataFrame) -> tuple[Any, float]:
        if not isinstance(y, pd.Series):
            raise ValueError("Beta likelihood requires univariate y")
        y = y.to_numpy(dtype = np.float_)
        logZ = 0.0
        return y, logZ

    def __call__(
        self, 
        y: npt.NDArray[np.float64],
        eta: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        alpha, dalpha = self.alpha_link.inv(eta[0])
        beta, dbeta = self.beta_link.inv(eta[1])
        log_y = np.log(y)
        log_1my = np.log1p(-y)
        f = np.sum(
            - special.betaln(alpha, beta)
            + ((alpha - 1) * log_y)
            + ((beta - 1) * log_1my)
        )
        dg_a = special.digamma(alpha)
        dg_b = special.digamma(beta)
        dg_ab = special.digamma(alpha + beta)
        galpha = dalpha * (dg_ab - dg_a + log_y)
        gbeta = dbeta * (dg_ab - dg_b + log_1my)
        g = np.stack([galpha, gbeta])
        tg_a = special.polygamma(1, alpha)
        tg_b = special.polygamma(1, beta)
        tg_ab = special.polygamma(1, alpha + beta)
        halpha = dalpha**2 * (tg_a - tg_ab)
        hbeta = dbeta**2 * (tg_b - tg_ab)
        halpha_beta = dalpha * dbeta * (- tg_ab)
        h = np.stack([[halpha, halpha_beta], [halpha_beta, hbeta]])
        return f, g, h

    def __repr__(self) -> str:
        return f"beta likelihood with alpha ({self.alpha_link}) and beta ({self.beta_link})"

def beta_canon(alpha_link: Link = log, beta_link: Link = log) -> Likelihood:
    return BetaCanon(alpha_link, beta_link)
