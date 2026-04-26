import numpy as np

class MarkowitzLQR:
    """
    Multi Asset Dynamic optimier 
    """
    def __init__(self, risk_aversion=2.0, tc_coeff=0.5):
        self.lam = risk_aversion
        self.gamma_tc = tc_coeff
        self.K_gains = None

    def solve(self, T, mu, Sigma):
        """
        solve_lqr_markowitz
        """
        n = len(mu)
        na = n + 1 
        
        # --- Matrix Construction ---
        A_aug = np.eye(na)
        B_aug = np.vstack([np.eye(n), np.zeros((1, n))])

        Q_aug = np.zeros((na, na))
        Q_aug[:n, :n] = 0.5 * self.lam * Sigma
        Q_aug[n, :n]  = -0.5 * mu
        Q_aug[:n, n]  = -0.5 * mu

        
        R_mat = 0.5 * self.gamma_tc * np.eye(n) + 0.5 * self.lam * Sigma

        M_aug = np.zeros((na, n))
        M_aug[:n, :] = 0.5 * self.lam * Sigma
        M_aug[n, :]  = -0.5 * mu

        
        Phi = np.zeros((na, na)) 
        K_gains = np.zeros((T, n, na))
        
        for t in range(T - 1, -1, -1):
            inner = R_mat + B_aug.T @ Phi @ B_aug
            K_t = np.linalg.solve(inner, M_aug.T + B_aug.T @ Phi @ A_aug)
            K_gains[t] = K_t
            Phi = Q_aug + A_aug.T @ Phi @ A_aug - K_t.T @ inner @ K_t

     
        self.K_gains = K_gains
        return self
        
        # Backward Induction
        for t in range(T - 1, -1, -1):
            inner = R_mat + B_aug.T @ Phi @ B_aug
            K_t = np.linalg.solve(inner, M_aug.T + B_aug.T @ Phi @ A_aug)
            self.K_gains[t] = K_t
            Phi = Q_aug + A_aug.T @ Phi @ A_aug - K_t.T @ inner @ K_t
        return self

  
    def predict(self, current_w, t):
        """
        Compute the optimal trade (delta weights) for a single time step.
        
        Args:
            current_w (array): Current portfolio weights (n,).
            t (int): Current time index.
            
        Returns:
            delta_w (array): Optimal rebalancing vector (n,).
        """
        if self.K_gains is None:
            raise ValueError("Model must be solved before prediction.")
            
        # Construct augmented state S_t = [w_{t-1}; 1]
        S_aug = np.append(current_w, 1.0)
        
        # Optimal feedback law: u_t = -K_t * S_t
        delta_w = -(self.K_gains[t] @ S_aug)
        
        return delta_w

                 

      
