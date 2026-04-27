"""
Unit tests for LQR Portfolio Optimizer
"""

import numpy as np
import pytest
from lqr_portfolio_optimizer import solve_lqr, execute_lqr


class TestLQRSolver:
    """Test suite for LQR solver functions."""
    
    def test_solve_lqr_output_shape(self):
        """Test that solve_lqr returns correct shape."""
        T, n, m = 10, 3, 1
        A = np.eye(n)
        B = np.ones((n, m))
        Q = np.eye(n) * 0.1
        R = np.eye(m) * 0.5
        M = np.zeros((n, m))
        
        K_gains = solve_lqr(T, A, B, Q, R, M)
        
        assert K_gains.shape == (T, m, n), f"Expected shape {(T, m, n)}, got {K_gains.shape}"
    
    def test_execute_lqr_output_shapes(self):
        """Test that execute_lqr returns correct shapes."""
        T, n, m = 10, 3, 1
        A = np.eye(n)
        B = np.ones((n, m))
        Q = np.eye(n) * 0.1
        R = np.eye(m) * 0.5
        M = np.zeros((n, m))
        s0 = np.array([1.0, 0.0, 0.0])
        
        K_gains = solve_lqr(T, A, B, Q, R, M)
        s_path, u_path = execute_lqr(T, A, B, K_gains, s0)
        
        assert s_path.shape == (T + 1, n), f"Expected s_path shape {(T+1, n)}, got {s_path.shape}"
        assert u_path.shape == (T, m), f"Expected u_path shape {(T, m)}, got {u_path.shape}"
    
    def test_initial_state_preserved(self):
        """Test that initial state is preserved in trajectory."""
        T, n, m = 5, 2, 1
        A = np.eye(n)
        B = np.ones((n, m))
        Q = np.eye(n) * 0.1
        R = np.eye(m) * 0.5
        M = np.zeros((n, m))
        s0 = np.array([1.5, -0.5])
        
        K_gains = solve_lqr(T, A, B, Q, R, M)
        s_path, u_path = execute_lqr(T, A, B, K_gains, s0)
        
        np.testing.assert_allclose(s_path[0], s0, rtol=1e-10,
                                   err_msg="Initial state not preserved")
    
    def test_dynamics_consistency(self):
        """Test that state trajectory follows dynamics."""
        T, n, m = 5, 2, 1
        A = np.array([[1.0, 0.1], [0.0, 0.9]])
        B = np.array([[1.0], [0.5]])
        Q = np.eye(n) * 0.1
        R = np.eye(m) * 0.5
        M = np.zeros((n, m))
        s0 = np.array([1.0, 0.0])
        
        K_gains = solve_lqr(T, A, B, Q, R, M)
        s_path, u_path = execute_lqr(T, A, B, K_gains, s0)
        
        # Check dynamics: s_{t+1} = A s_t + B u_t
        for t in range(T):
            s_next_expected = A @ s_path[t] + B @ u_path[t]
            np.testing.assert_allclose(
                s_path[t + 1], s_next_expected, rtol=1e-10,
                err_msg=f"Dynamics violated at step {t}"
            )
    
    def test_optimal_execution_problem(self):
        """Test on a standard optimal execution problem."""
        T = 20
        gamma = 1.0
        sigma_sq = 0.04
        eta = 0.1
        rho = 0.95
        beta = 0.8
        alpha0 = 1.0
        
        A = np.array([[1, 0, 0], [0, rho, 0], [0, 0, beta]], dtype=float)
        B = np.array([[1], [0], [beta * eta]], dtype=float)
        Q = np.diag([0.5 * gamma * sigma_sq, 0, 0])
        R = np.array([[0.5 * gamma * sigma_sq + 0.5 * eta]])
        M = np.array([[0.5 * gamma * sigma_sq], [-0.5], [0.5]])
        s0 = np.array([0.0, alpha0, 0.0])
        
        K_gains = solve_lqr(T, A, B, Q, R, M)
        s_path, u_path = execute_lqr(T, A, B, K_gains, s0)
        
        # Basic sanity checks
        assert K_gains.shape == (T, 1, 3)
        assert s_path.shape == (T + 1, 3)
        assert u_path.shape == (T, 1)
        
        # Check initial state
        np.testing.assert_allclose(s_path[0], s0, rtol=1e-10)
    
    def test_markowitz_problem(self):
        """Test on a multi-asset Markowitz problem."""
        n_assets = 3
        T = 10
        lam = 1.0
        gamma_tc = 0.1
        
        # Simple covariance and returns
        Sigma = np.eye(n_assets) * 0.04 + np.ones((n_assets, n_assets)) * 0.01
        mu = np.array([0.005, 0.007, 0.006])
        w0 = np.array([0.3, 0.4, 0.3])
        
        n_aug = n_assets + 1
        A = np.eye(n_aug)
        B = np.vstack([np.eye(n_assets), np.zeros((1, n_assets))])
        
        Q = np.zeros((n_aug, n_aug))
        Q[:n_assets, :n_assets] = 0.5 * lam * Sigma
        Q[n_aug-1, :n_assets] = -0.5 * mu
        Q[:n_assets, n_aug-1] = -0.5 * mu
        
        R = 0.5 * gamma_tc * np.eye(n_assets) + 0.5 * lam * Sigma
        
        M = np.zeros((n_aug, n_assets))
        M[:n_assets, :] = 0.5 * lam * Sigma
        M[n_aug-1, :] = -0.5 * mu
        
        s0 = np.append(w0, 1.0)
        
        K_gains = solve_lqr(T, A, B, Q, R, M)
        s_path, u_path = execute_lqr(T, A, B, K_gains, s0)
        
        # Basic sanity checks
        assert K_gains.shape == (T, n_assets, n_aug)
        assert s_path.shape == (T + 1, n_aug)
        assert u_path.shape == (T, n_assets)
        
        # Check that constant term stays 1
        np.testing.assert_allclose(s_path[:, -1], 1.0, rtol=1e-10,
                                   err_msg="Constant term should stay 1")
    
    def test_zero_horizon(self):
        """Test that zero horizon returns empty gains."""
        T = 0
        n, m = 2, 1
        A = np.eye(n)
        B = np.ones((n, m))
        Q = np.eye(n) * 0.1
        R = np.eye(m) * 0.5
        M = np.zeros((n, m))
        
        K_gains = solve_lqr(T, A, B, Q, R, M)
        
        assert K_gains.shape == (0, m, n)
    
    def test_single_step_horizon(self):
        """Test single time step problem."""
        T = 1
        n, m = 2, 1
        A = np.eye(n)
        B = np.ones((n, m))
        Q = np.eye(n) * 0.1
        R = np.eye(m) * 0.5
        M = np.zeros((n, m))
        s0 = np.array([1.0, 0.0])
        
        K_gains = solve_lqr(T, A, B, Q, R, M)
        s_path, u_path = execute_lqr(T, A, B, K_gains, s0)
        
        assert K_gains.shape == (1, m, n)
        assert s_path.shape == (2, n)
        assert u_path.shape == (1, m)


class TestInputValidation:
    """Test input validation and edge cases."""
    
    def test_mismatched_dimensions(self):
        """Test that mismatched dimensions raise errors."""
        T = 10
        A = np.eye(3)
        B = np.ones((2, 1))  # Wrong dimension
        Q = np.eye(3)
        R = np.eye(1)
        M = np.zeros((3, 1))
        
        with pytest.raises((ValueError, IndexError)):
            solve_lqr(T, A, B, Q, R, M)
    
    def test_non_square_state_matrix(self):
        """Test that non-square A matrix causes issues."""
        T = 10
        A = np.ones((3, 2))  # Not square
        B = np.ones((3, 1))
        Q = np.eye(3)
        R = np.eye(1)
        M = np.zeros((3, 1))
        
        # May raise error or produce invalid results
        try:
            K_gains = solve_lqr(T, A, B, Q, R, M)
            # If it doesn't raise, dimensions should still be wrong
            assert K_gains.shape[2] != A.shape[0], "Should have dimension mismatch"
        except (ValueError, IndexError, np.linalg.LinAlgError):
            pass  # Expected to fail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
