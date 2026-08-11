"""Primal-space Sylvester equation solver (PyTorch).

D-CREM solves the Sylvester equation in primal space (d' × q) instead of
CREM's dual space (N × q).  This is the key computational advantage:
d' (128) ≪ N (2500–5000), so each solve is ~1000× faster.

  Sy_A V + V Sy_B = Sy_C

  Sy_A ∈ R^{d'×d'}, Sy_B ∈ R^{q×q}, Sy_C ∈ R^{d'×q}

Solved via Kronecker vectorisation:
  (I_q ⊗ Sy_A + Sy_Bᵀ ⊗ I_{d'}) vec(V) = vec(Sy_C)

When d'·q ≤ 5000 the direct Kronecker solve is fastest.  For larger systems
we fall back to the Bartels-Stewart algorithm via scipy.
"""

import torch


def solve_sylvester(Sy_A, Sy_B, Sy_C):
    """Solve Sy_A V + V Sy_B = Sy_C for V.

    Parameters
    ----------
    Sy_A : (d, d) tensor
    Sy_B : (q, q) tensor
    Sy_C : (d, q) tensor

    Returns
    -------
    V : (d, q) tensor
    """
    d, q = Sy_C.shape
    device, dtype = Sy_A.device, Sy_A.dtype
    n_total = d * q

    # Small system: use Kronecker form directly
    if n_total <= 5000:
        # I_q ⊗ Sy_A   (must be contiguous for torch.kron)
        I_q = torch.eye(q, device=device, dtype=dtype)
        I_d = torch.eye(d, device=device, dtype=dtype)
        K_A = torch.kron(I_q, Sy_A.contiguous())
        K_B = torch.kron(Sy_B.T.contiguous(), I_d)
        M = K_A + K_B
        c_vec = Sy_C.T.reshape(-1, 1)            # vec(C), column-major → (q·d, 1)
        v_vec = torch.linalg.solve(M, c_vec)
        V = v_vec.reshape(q, d).T                 # (d, q)
        return V

    # Large system: fall back to scipy Bartels-Stewart
    return _solve_sylvester_scipy(Sy_A, Sy_B, Sy_C)


def _solve_sylvester_scipy(Sy_A, Sy_B, Sy_C):
    """Fallback: use scipy's solve_sylvester (Bartels-Stewart algorithm)."""
    from scipy.linalg import solve_sylvester as scipy_sylvester
    device = Sy_A.device
    A_np = Sy_A.detach().cpu().numpy()
    B_np = Sy_B.detach().cpu().numpy()
    C_np = Sy_C.detach().cpu().numpy()
    V_np = scipy_sylvester(A_np, B_np, C_np)
    return torch.as_tensor(V_np, dtype=Sy_C.dtype, device=device)
