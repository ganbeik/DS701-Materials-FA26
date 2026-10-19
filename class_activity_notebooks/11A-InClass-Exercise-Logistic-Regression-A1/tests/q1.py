from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 2

@test_case(points=None, hidden=False)
def test_sigmoid_and_ll_basics(sigmoid, log_likelihood, ll_init, X_design, y_cls, n_pts):
    import numpy as np
    assert abs(sigmoid(0.0) - 0.5) < 1e-12
    assert np.allclose(sigmoid(np.array([-100.0, 100.0])), [0.0, 1.0])
    assert np.allclose(sigmoid(np.array([1.0, -1.0])), [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
    assert isinstance(ll_init, float)
    assert abs(log_likelihood(np.zeros(3), X_design, y_cls) - n_pts * np.log(0.5)) < 1e-09, 'at beta = 0 every p_i is 0.5'
    assert ll_init < 0

