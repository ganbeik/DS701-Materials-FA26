from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 2

@test_case(points=None, hidden=False)
def test_sklearn_fit_basics(clf_pts, beta_sk, max_abs_diff, boundary_slope, boundary_intercept, train_acc):
    import numpy as np
    assert np.asarray(beta_sk).shape == (3,), 'beta_sk = [intercept, coef_1, coef_2]'
    assert np.allclose(beta_sk[1:], clf_pts.coef_[0]) and abs(beta_sk[0] - clf_pts.intercept_[0]) < 1e-12
    assert max_abs_diff < 0.05, 'gradient ascent and sklearn should agree closely -- did the loop run to convergence?'
    x1 = 0.37
    x2 = boundary_slope * x1 + boundary_intercept
    assert abs(beta_sk[0] + beta_sk[1] * x1 + beta_sk[2] * x2) < 1e-08, 'the boundary is beta0 + beta1 x1 + beta2 x2 = 0'
    assert 0.5 < train_acc <= 1.0

