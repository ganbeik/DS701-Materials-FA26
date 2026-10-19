from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 3

@test_case(points=None, hidden=False)
def test_gradient_basics(grad_log_likelihood, beta_step1, ll_step1, ll_init, beta_ga, X_design, y_cls):
    import numpy as np
    g0 = grad_log_likelihood(np.zeros(3), X_design, y_cls)
    assert np.asarray(g0).shape == (3,), 'gradient must be a length-3 array'
    assert np.allclose(g0, X_design.T @ (y_cls - 0.5)), 'check the formula X^T (y - p)'
    assert ll_step1 > ll_init, 'one gradient-ascent step must increase the log-likelihood'
    assert np.asarray(beta_step1).shape == (3,) and np.asarray(beta_ga).shape == (3,)

