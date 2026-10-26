from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 0

@test_case(points=None, hidden=False)
def test_svd_shapes(U, s, Vt, A_rebuilt, sigma_max, rank_A, A, m, n):
    import numpy as np
    assert U.shape == (m, n), f'U must be (m, n) = ({m}, {n}); did you pass full_matrices=False?'
    assert s.shape == (n,), 's must be a 1-D array of n singular values'
    assert Vt.shape == (n, n)
    assert np.all(s >= 0) and np.all(np.diff(s) <= 1e-09), 'singular values must be non-negative and sorted largest first'
    assert np.allclose(U.T @ U, np.eye(n), atol=1e-08), 'columns of U must be orthonormal'
    assert np.allclose(A_rebuilt, A, atol=1e-06), 'U @ diag(s) @ Vt must reproduce A'
    assert isinstance(sigma_max, float) and abs(sigma_max - s[0]) < 1e-09
    assert isinstance(rank_A, int) and 1 <= rank_A <= min(m, n)

