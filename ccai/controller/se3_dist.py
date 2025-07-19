import numpy as np

def skew(v):
    """Returns the skew-symmetric matrix of a 3-vector."""
    return np.array([
        [ 0,   -v[2],  v[1]],
        [ v[2],  0,   -v[0]],
        [-v[1], v[0],   0 ]
    ])

def log_SO3(R):
    """Logarithm map for SO(3) (returns 3-vector)."""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if np.isclose(theta, 0):
        return np.zeros(3)
    lnR = theta / (2 * np.sin(theta)) * (R - R.T)
    return np.array([lnR[2,1], lnR[0,2], lnR[1,0]])

def log_SE3(T):
    """Logarithm map for SE(3), returns a 6D vector [omega, v]."""
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

def vee_se3(xi_mat):
    """Converts a 4x4 matrix in se(3) to a 6D vector"""
    omega = np.array([xi_mat[2,1], xi_mat[0,2], xi_mat[1,0]])
    v = xi_mat[:3, 3]
    return np.hstack([omega, v])

def adjoint_SE3(T):
    """Adjoint representation of SE(3). Returns a 6x6 matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    t_skew = skew(t)
    upper = np.hstack((R, np.zeros((3,3))))
    lower = np.hstack((t_skew @ R, R))
    return np.vstack((upper, lower))

def inverse_SE3(T):
    """Inverse of an SE(3) matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def se3_distance(T1, T2, W_P):
    """SE(3) distance: ||log(T1^{-1} T2)||"""
    T_delta = inverse_SE3(T1) @ T2
    xi = log_SE3(T_delta)
    
    # W_P projection
    xi_proj = xi @ W_P
    return np.linalg.norm(xi_proj), xi_proj, T_delta

def se3_distance_gradient(T1, T2, W_P):
    """Returns the gradient of the SE(3) distance with respect to T1."""
    dist, xi, T_delta = se3_distance(T1, T2, W_P)
    if np.isclose(dist, 0):
        return np.zeros((6,6)), xi, T_delta  # Gradient is zero at the minimum
    Ad = adjoint_SE3(T_delta)
    grad = -xi / dist @ Ad  # Outer product in the tangent space
    grad_proj = grad @ W_P
    return dist, grad_proj

# ---------- Example Usage ----------

if __name__ == "__main__":
    # Create two random SE(3) transformations
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

    dist, xi, T_delta = se3_distance(T1, T2)
    dist, grad = se3_distance_gradient(T1, T2, np.eye(6))

    print(f"SE(3) distance: {dist:.4f}")
    print(f"Log-vector xi: {xi}")
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient matrix:\n{grad}")