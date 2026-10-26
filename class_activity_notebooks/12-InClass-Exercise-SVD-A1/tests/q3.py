from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 0

@test_case(points=None, hidden=False)
def test_rank_k_approx_basic(rank_k_approx, A_k, KS, m, n):
    import numpy as np
    U_t = np.eye(3)[:, :2]
    s_t = np.array([3.0, 1.0])
    Vt_t = np.eye(2)
    A1_t = rank_k_approx(U_t, s_t, Vt_t, 1)
    assert A1_t.shape == (3, 2)
    assert np.allclose(A1_t, np.array([[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]])), 'rank-1 truncation of diag(3,1) must keep only the 3'
    assert np.allclose(rank_k_approx(U_t, s_t, Vt_t, 2), np.array([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    assert list(A_k) == KS
    for k in KS:
        assert A_k[k].shape == (m, n), f'A_k[{k}] must be {m} x {n}'
        assert np.linalg.matrix_rank(A_k[k]) == k, f'A_k[{k}] must have rank exactly {k}'

