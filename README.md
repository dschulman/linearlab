# linearlab

Some playground/experimental code implementing as wide a variety as possible of linear models, generalized in many ways.
Emphasis is on clean and flexible code rather than performance.

This is (very roughly) following a statsmodel-style API:
* Every model corresponds to a class.
    - The constructor accepts raw design matrices.
    - A `from_formula()` static method accepts R-style formulas.
* Calling `fit()` on a model instance produces a `Fit` object.

Nothing in here should be considered ready for real work.
Assume it's untested, and breaks silently in surprising ways.

A list of planned features, in no particular order:
* An extensive set of likelihood models and link functions. Not restricted to exponential dispersion models.
* Multi-parameter likelihood models, allowing e.g. VGLM-like models (location-scale-shape).
* Flexible regularization penalties applied to most models (maybe not random/mixed-effects?).
    - At least supporting ridge, lasso, and elastic net.
    - (maybe?) implicit gradient methods on CV error for choosing penalties.
    - (maybe?) implicit gradient methods with infinitesimal jackknife (or other LOOCV approximation).
    - (maybe?) approximate marginal likelihood optimization of penalties.
* A generic mixed model implementation, with arbitrary random effect design matrices and covariance structures.
    - Reasonable efficiency via sparse design matrices and Kronecker-factored covariance
    - Laplace approximation for GLMMs.
    - (maybe?) generic adaptive Gauss-Hermite quadrature for any model, handled by treating the random effect design matrix as a sparse bipartite graph.
    - (maybe?) expectation propagation, also using the bipartite graph approach. Potentially a very nice approach especially for binary data.
