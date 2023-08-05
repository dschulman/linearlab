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
    return f, theta
