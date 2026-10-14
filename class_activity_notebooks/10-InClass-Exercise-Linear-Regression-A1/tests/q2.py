from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 0

@test_case(points=None, hidden=False)
def test_projection_properties(P, y_hat_P, resid, max_abs_xt_e, P_trace, sym_err, idem_err, X1, y, n):
    import numpy as np
    assert np.asarray(P).shape == (n, n), 'P must be n x n'
    assert np.allclose(y_hat_P, X1 @ np.linalg.lstsq(X1, y, rcond=None)[0], atol=1e-06), 'P @ y must equal the least-squares fitted values'
    assert np.allclose(resid, y - y_hat_P)
    assert max_abs_xt_e < 1e-05, f'residuals must be orthogonal to the columns of X (got {max_abs_xt_e:.2e})'
    assert abs(P_trace - X1.shape[1]) < 1e-06, 'trace of the projection = number of columns of X'
    assert sym_err < 1e-08 and idem_err < 1e-08, 'P must be symmetric and idempotent'

@test_case(points=None, hidden=False)
def test_leverage_basic(leverage, mean_leverage, X1, n):
    import numpy as np
    assert np.asarray(leverage).shape == (n,)
    assert np.all(leverage >= -1e-12) and np.all(leverage <= 1 + 1e-12), 'leverages lie in [0, 1]'
    assert abs(mean_leverage - X1.shape[1] / n) < 1e-09, 'mean leverage is p / n'

