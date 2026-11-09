from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 0

@test_case(points=None, hidden=False)
def test_scaled_shape(X_scaled, X):
    assert X_scaled is not None, 'X_scaled is not defined'
    assert X_scaled.shape == X.shape, 'X_scaled must have the same shape as X'

@test_case(points=None, hidden=False)
def test_scaled_moments(X_scaled):
    import numpy as np
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-08), 'every column of X_scaled must have mean 0'
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-08), 'every column of X_scaled must have std 1'

