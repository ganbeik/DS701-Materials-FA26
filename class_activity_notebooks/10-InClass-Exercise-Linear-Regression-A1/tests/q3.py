from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 0

@test_case(points=None, hidden=False)
def test_two_feature_shapes(X2, beta_2, r2_2, r2_1, pred2_bmi30_s5, bmi_coef_change, n):
    import numpy as np
    assert np.asarray(X2).shape == (n, 3) and np.allclose(np.asarray(X2)[:, 0], 1.0), 'X2 must be [1, bmi, s5]'
    assert np.asarray(beta_2).shape == (3,)
    assert r2_1 <= r2_2 <= 1.0, 'adding a column can never lower the in-sample R^2'
    assert isinstance(pred2_bmi30_s5, float) and isinstance(bmi_coef_change, float)

