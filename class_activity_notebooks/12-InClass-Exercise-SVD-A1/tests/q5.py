from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 0

@test_case(points=None, hidden=False)
def test_eckart_young_basic(B_cols, err_random, err_cols, err_svd, eckart_young_holds, cols_penalty_pct, A, cols, k_test):
    import numpy as np
    assert B_cols.shape == A.shape
    assert np.linalg.matrix_rank(B_cols) <= k_test, 'B_cols must have rank at most k_test'
    assert np.allclose(B_cols[:, cols], A[:, cols], atol=1e-06), 'the chosen columns should be reproduced exactly by their own span'
    assert all((isinstance(v, float) and v > 0 for v in (err_random, err_cols, err_svd)))
    assert eckart_young_holds is True, 'the truncated SVD must beat both competitors'
    assert cols_penalty_pct > 0

