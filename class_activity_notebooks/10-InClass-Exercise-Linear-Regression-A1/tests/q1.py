from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 0

@test_case(points=None, hidden=False)
def test_design_matrix(X1, n):
    import numpy as np
    X1 = np.asarray(X1)
    assert X1.shape == (n, 2), f'X1 must be (n, 2), got {X1.shape}'
    assert np.allclose(X1[:, 0], 1.0), 'first column of X1 must be all ones (the intercept)'

@test_case(points=None, hidden=False)
def test_three_solvers_agree(beta_ne, beta_lstsq, beta_sk, r2_1, pred_bmi30):
    import numpy as np
    for b in (beta_ne, beta_lstsq, beta_sk):
        assert np.asarray(b).shape == (2,), 'each beta must be a length-2 array [intercept, slope]'
    assert np.allclose(beta_ne, beta_lstsq, atol=1e-06), 'normal equations and lstsq disagree'
    assert np.allclose(beta_ne, beta_sk, atol=1e-06), 'normal equations and sklearn disagree'
    assert 0.0 <= r2_1 <= 1.0
    assert isinstance(pred_bmi30, float)

