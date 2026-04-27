"""
src  —  DP/LQR solvers for portfolio optimization

Submodules
----------
lqr          Generic finite-horizon LQR (backward induction)
execution    Single-asset optimal execution
markowitz    Multi-asset Markowitz mean-variance
risk_parity  Risk-parity target tracking

Quick start
-----------
>>> from src import lqr, execution, markowitz, risk_parity

Generic LQR:
>>> K = lqr.solve(T, A, B, Q, R, M)
>>> s_path, u_path = lqr.simulate(T, A, B, K, s0)

Execution problem:
>>> K = execution.solve(T, gamma=1.0, sigma_sq=0.04, eta=0.1, rho=0.95, beta=0.8)
>>> w_path = execution.simulate(T, K, initial_alpha=0.5, eta=0.1, rho=0.95, beta=0.8)

Markowitz:
>>> K = markowitz.solve(T, mu, Sigma, lam=2.0, gamma_tc=0.5)
>>> w_path = markowitz.simulate(T, K, w0)
>>> val = markowitz.objective(w_path, mu, Sigma, lam=2.0, gamma_tc=0.5)

Risk parity:
>>> w_RP = risk_parity.target(Sigma)
>>> K = risk_parity.solve(T, w_RP, kappa=2.0, gamma_tc=0.5)
>>> w_path = risk_parity.simulate(T, K, w0)
>>> c = risk_parity.cost(w_path, w_RP, kappa=2.0, gamma_tc=0.5)
"""

from . import lqr, execution, markowitz, risk_parity

__all__ = ["lqr", "execution", "markowitz", "risk_parity"]
