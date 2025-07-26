import numpy as np

def skew(v):
    """Returns the skew-symmetric matrix of a 3-vector or batch of 3-vectors."""
    if v.ndim == 1:
        # Single vector case
        return np.array([
            [ 0,   -v[2],  v[1]],
            [ v[2],  0,   -v[0]],
            [-v[1], v[0],   0 ]
        ])
    else:
        # Batch case: v has shape (B, 3)
        B = v.shape[0]
        skew_mats = np.zeros((B, 3, 3))
        skew_mats[:, 0, 1] = -v[:, 2]
        skew_mats[:, 0, 2] = v[:, 1]
        skew_mats[:, 1, 0] = v[:, 2]
        skew_mats[:, 1, 2] = -v[:, 0]
        skew_mats[:, 2, 0] = -v[:, 1]
        skew_mats[:, 2, 1] = v[:, 0]
        return skew_mats

def log_SO3(R):
    """Logarithm map for SO(3) (returns 3-vector or batch of 3-vectors)."""
    if R.ndim == 2:
        # Single matrix case
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if np.isclose(theta, 0):
            return np.zeros(3)
        lnR = theta / (2 * np.sin(theta)) * (R - R.T)
        return np.array([lnR[2,1], lnR[0,2], lnR[1,0]])
    else:
        # Batch case: R has shape (B, 3, 3)
        B = R.shape[0]
        trace_R = np.trace(R, axis1=1, axis2=2)  # (B,)
        theta = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))  # (B,)
        
        result = np.zeros((B, 3))
        non_zero_mask = ~np.isclose(theta, 0)
        
        if np.any(non_zero_mask):
            theta_nz = theta[non_zero_mask]
            R_nz = R[non_zero_mask]
            lnR = (theta_nz / (2 * np.sin(theta_nz)))[:, None, None] * (R_nz - R_nz.transpose(0, 2, 1))
            result[non_zero_mask, 0] = lnR[:, 2, 1]
            result[non_zero_mask, 1] = lnR[:, 0, 2]
            result[non_zero_mask, 2] = lnR[:, 1, 0]
        
        return result

def log_SE3(T):
    """Logarithm map for SE(3), returns a 6D vector [omega, v] or batch of 6D vectors."""
    if T.ndim == 2:
        # Single matrix case
        R = T[:3, :3]
        t = T[:3, 3]
        omega = log_SO3(R)
        theta = np.linalg.norm(omega)
        if np.isclose(theta, 0):
            V_inv = np.eye(3)
        else:
            omega_hat = skew(omega / theta)
            V = (np.eye(3)
                 + (1 - np.cos(theta)) / theta**2 * omega_hat
                 + (theta - np.sin(theta)) / theta**3 * (omega_hat @ omega_hat))
            V_inv = np.linalg.inv(V)
        v = V_inv @ t
        return np.hstack([omega, v])
    else:
        # Batch case: T has shape (B, 4, 4)
        B = T.shape[0]
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        omega = log_SO3(R)  # (B, 3)
        theta = np.linalg.norm(omega, axis=1)  # (B,)
        
        result = np.zeros((B, 6))
        
        # Handle zero rotation case
        zero_mask = np.isclose(theta, 0)
        if np.any(zero_mask):
            result[zero_mask, :3] = omega[zero_mask]
            result[zero_mask, 3:] = t[zero_mask]
        
        # Handle non-zero rotation case
        non_zero_mask = ~zero_mask
        if np.any(non_zero_mask):
            theta_nz = theta[non_zero_mask]
            omega_nz = omega[non_zero_mask]
            t_nz = t[non_zero_mask]
            
            omega_hat = skew(omega_nz / theta_nz[:, None])  # (B_nz, 3, 3)
            eye = np.eye(3)[None, :, :].repeat(np.sum(non_zero_mask), axis=0)
            
            V = (eye
                 + ((1 - np.cos(theta_nz)) / theta_nz**2)[:, None, None] * omega_hat
                 + ((theta_nz - np.sin(theta_nz)) / theta_nz**3)[:, None, None] * np.einsum('bij,bjk->bik', omega_hat, omega_hat))
            
            V_inv = np.linalg.inv(V)
            v_nz = np.einsum('bij,bj->bi', V_inv, t_nz)
            
            result[non_zero_mask, :3] = omega_nz
            result[non_zero_mask, 3:] = v_nz
        
        return result

def vee_se3(xi_mat):
    """Converts a 4x4 matrix in se(3) to a 6D vector or batch of 6D vectors"""
    if xi_mat.ndim == 2:
        # Single matrix case
        omega = np.array([xi_mat[2,1], xi_mat[0,2], xi_mat[1,0]])
        v = xi_mat[:3, 3]
        return np.hstack([omega, v])
    else:
        # Batch case: xi_mat has shape (B, 4, 4)
        B = xi_mat.shape[0]
        omega = np.stack([xi_mat[:, 2, 1], xi_mat[:, 0, 2], xi_mat[:, 1, 0]], axis=1)
        v = xi_mat[:, :3, 3]
        return np.concatenate([omega, v], axis=1)

def adjoint_SE3(T):
    """Adjoint representation of SE(3). Returns a 6x6 matrix or batch of 6x6 matrices."""
    if T.ndim == 2:
        # Single matrix case
        R = T[:3, :3]
        t = T[:3, 3]
        t_skew = skew(t)
        upper = np.hstack((R, np.zeros((3,3))))
        lower = np.hstack((t_skew @ R, R))
        return np.vstack((upper, lower))
    else:
        # Batch case: T has shape (B, 4, 4)
        B = T.shape[0]
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        t_skew = skew(t)  # (B, 3, 3)
        
        adj = np.zeros((B, 6, 6))
        adj[:, :3, :3] = R
        adj[:, :3, 3:] = 0
        adj[:, 3:, :3] = np.einsum('bij,bjk->bik', t_skew, R)
        adj[:, 3:, 3:] = R
        
        return adj

def inverse_SE3(T):
    """Inverse of an SE(3) matrix."""
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    T_inv = np.eye(4).reshape(1, 4, 4).repeat(T.shape[0], axis=0)
    T_inv[:, :3, :3] = R.transpose(0, 2, 1)
    T_inv[:, :3, 3] = np.einsum('bij,bj->bi', -R.transpose(0, 2, 1), t)
    return T_inv

def se3_distance(T1, T2, W_P):
    """SE(3) distance: ||log(T1^{-1} T2)|| for single matrices or batches"""
    if T1.ndim == 2:
        # Single matrix case
        T_delta = np.linalg.inv(T1) @ T2
        xi = log_SE3(T_delta)
        
        # W_P projection
        xi_proj = xi @ W_P
        return np.linalg.norm(xi_proj), xi_proj, T_delta
    else:
        # Batch case: T1, T2 have shape (B, 4, 4)
        T_delta = np.einsum('bij,bjk->bik', inverse_SE3(T1), T2)
        xi = log_SE3(T_delta).flatten()  # (B, 6)
        
        # W_P projection
        xi_proj = xi @ W_P # (B, W_P.shape[1])
        xi_proj = xi_proj.reshape(T_delta.shape[0], -1)
        dist = np.linalg.norm(xi_proj, axis=1)  # (B,)
        return dist, xi_proj, T_delta

def se3_distance_gradient(T1, T2, W_P):
    """Returns the gradient of the SE(3) distance with respect to T1."""
    if T1.ndim == 2:
        # Single matrix case
        dist, xi_proj, T_delta = se3_distance(T1, T2, W_P)
        # Get the unprojected xi for gradient computation
        xi = log_SE3(T_delta)
        # if np.isclose(dist, 0):
        #     return np.zeros((6,6)), xi, T_delta  # Gradient is zero at the minimum
        Ad = adjoint_SE3(T_delta)
        grad = -xi / dist @ Ad  # Gradient in the tangent space
        grad_proj = grad @ W_P
        return dist, grad_proj
    else:
        # Batch case: T1, T2 have shape (B, 4, 4)
        dist, xi_proj, T_delta = se3_distance(T1, T2, W_P)
        # Get the unprojected xi for gradient computation
        xi = log_SE3(T_delta)  # (B, 6)
        # Handle case where distance is zero
        zero_mask = np.isclose(dist, 0)
        
        Ad = adjoint_SE3(T_delta)  # (B, 6, 6)
        
        # Compute gradient for non-zero distances
        grad_proj = np.zeros((T1.shape[0], xi_proj.shape[1]))
        non_zero_mask = ~zero_mask
        
        if np.any(non_zero_mask):
            xi_nz = xi[non_zero_mask]
            dist_nz = dist[non_zero_mask]
            Ad_nz = Ad[non_zero_mask]
            
            # grad = -xi / dist @ Ad for each batch element
            grad_nz = -np.einsum('bi,b,bij->bj', xi_nz, 1.0 / dist_nz, Ad_nz)
            grad_proj_nz = (grad_nz.flatten() @ W_P).reshape(T_delta.shape[0], -1)
            grad_proj[non_zero_mask] = grad_proj_nz
        
        return dist, grad_proj

# ---------- Example Usage ----------

if __name__ == "__main__":
    # Create two random SE(3) transformations for single matrix test
    np.random.seed(42)
    R1 = np.eye(3)
    t1 = np.array([0, 0, 0])
    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, 3] = t1

    angle = np.pi / 6
    R2 = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    t2 = np.array([1, 2, 3])
    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3] = t2

    W_P = np.eye(6)
    
    # Test single matrix case
    print("=== Single Matrix Case ===")
    dist, xi_proj, T_delta = se3_distance(T1, T2, W_P)
    dist_grad, grad = se3_distance_gradient(T1, T2, W_P)

    print(f"SE(3) distance: {dist:.4f}")
    print(f"Log-vector xi_proj: {xi_proj}")
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient matrix:\n{grad}")

    # Test batched case
    print("\n=== Batched Case ===")
    batch_size = 3
    T1_batch = np.tile(T1[None, :, :], (batch_size, 1, 1))
    T2_batch = np.tile(T2[None, :, :], (batch_size, 1, 1))
    
    # Add some noise to make them different
    T1_batch[1, :3, 3] += np.array([0.1, 0.2, 0.3])
    T2_batch[2, :3, 3] += np.array([0.5, 0.1, 0.2])

    dist_batch, xi_proj_batch, T_delta_batch = se3_distance(T1_batch, T2_batch, W_P)
    dist_grad_batch, grad_batch = se3_distance_gradient(T1_batch, T2_batch, W_P)

    print(f"Batched SE(3) distances: {dist_batch}")
    print(f"Batched xi_proj shape: {xi_proj_batch.shape}")
    print(f"Batched gradient shape: {grad_batch.shape}")
    print(f"First batch gradient:\n{grad_batch[0]}")
    
    # Verify single vs batch consistency for first element
    print(f"\nConsistency check:")
    print(f"Single distance: {dist:.6f}, Batch[0] distance: {dist_batch[0]:.6f}")
    print(f"Distance diff: {abs(dist - dist_batch[0]):.8f}")
    print(f"Gradient diff norm: {np.linalg.norm(grad - grad_batch[0]):.8f}")