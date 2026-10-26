from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 0

@test_case(points=None, hidden=False)
def test_refresher_shapes(eig_AtA, sigma_sq_rel_diff, sigma_sq_matches, s_toy, rank_toy, n_tiny_toy, M_toy, n):
    import numpy as np
    assert eig_AtA.shape == (n,), 'one eigenvalue per column of A'
    assert np.all(np.diff(eig_AtA) <= 1e-06), 'eig_AtA must be sorted descending'
    assert isinstance(sigma_sq_rel_diff, float) and sigma_sq_rel_diff >= 0
    assert isinstance(sigma_sq_matches, bool)
    assert s_toy.shape == (min(M_toy.shape),)
    assert isinstance(rank_toy, int) and isinstance(n_tiny_toy, int)
    assert rank_toy + n_tiny_toy == len(s_toy)
    assert rank_toy == np.linalg.matrix_rank(M_toy), 'your threshold count should agree with np.linalg.matrix_rank here'

