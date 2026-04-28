"""
LQR Portfolio Optimizer
=======================

A fast Dynamic Programming-based solver for multi-period portfolio optimization problems.

Main Functions
--------------
solve_lqr : Compute optimal feedback gains via backward induction
execute_lqr : Forward simulation using pre-computed gains
solve_and_execute_lqr : Convenience function combining both steps

Examples
--------
>>> from lqr_portfolio_optimizer import solve_lqr, execute_lqr
>>> import numpy as np
>>> 
>>> # Define problem parameters
>>> T = 20
>>> A = np.array([[1, 0, 0], [0, 0.95, 0], [0, 0, 0.8]])
>>> B = np.array([[1], [0], [0.08]])
>>> Q = np.diag([0.02, 0, 0])
>>> R = np.array([[0.07]])
>>> M = np.array([[0.02], [-0.5], [0.5]])
>>> s0 = np.array([0.0, 1.0, 0.0])
>>> 
>>> # Solve
>>> K_gains = solve_lqr(T, A, B, Q, R, M)
>>> s_path, u_path = execute_lqr(T, A, B, K_gains, s0)
"""

from .solver import solve_lqr, execute_lqr, solve_and_execute_lqr, LQRResult

__version__ = "0.1.0"
__author__ = "LQR Portfolio Optimizer Contributors"
__all__ = ["solve_lqr", "execute_lqr", "solve_and_execute_lqr", "LQRResult"]
