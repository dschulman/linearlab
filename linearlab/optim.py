from collections.abc import Callable
import numpy as np
import numpy.linalg as npl
import numpy.typing as npt

def newton_maxlik(
    fun: Callable[[npt.NDArray[np.float64]], tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    theta0: npt.NDArray[np.float64],
    tol: float = 1e-8,
    max_iter: int = 1000
) -> npt.NDArray[np.float64]:
    theta = theta0
    f, g, h = fun(theta0)
    ng = npl.solve(h, g)
    step = 1.0
    for k in range(max_iter):
        theta1 = theta + (step * ng)
        f1, g1, h1 = fun(theta1)
        if np.linalg.norm(theta1 - theta, np.inf) <= tol:
            break
        if f1 > f:
            theta = theta1
            f, g, h = f1, g1, h1
            ng = npl.solve(h, g)
            step = 1.0
        else:
            step = 0.5 * step
    return theta, f, g, h


def soft_threshold(
    k: float | npt.NDArray[np.float64], 
    x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.maximum(0, x - k) - np.maximum(0, -x - k)

# Based on Section 6.3 of: https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
def l1_admm(
    f: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], float], npt.NDArray[np.float64]],
    n: int,
    l1_reg: float | npt.NDArray[np.float64],
    rho: float = 1.0,
    tol_abs: float = 1e-4,
    tol_rel: float = 1e-2,
    max_iter: int = 1000,
) -> npt.NDArray[np.float64]:
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    tol_abs *= np.sqrt(n)
    for k in range(max_iter):
        x_new = f(x, z-u, rho)
        z_new = soft_threshold(l1_reg / rho, x_new + u)
        u_new = u + x_new - z_new
        r_norm = npl.norm(x_new - z_new)
        s_norm = npl.norm(-rho * (z_new - z))
        eps_pri = tol_abs + tol_rel * max(npl.norm(x_new), npl.norm(z_new))
        eps_dual = tol_abs + tol_rel * npl.norm(rho * u_new)
        x = x_new
        z = z_new
        u = u_new
        if (r_norm < eps_pri) and (s_norm < eps_dual):
            break
    return x
