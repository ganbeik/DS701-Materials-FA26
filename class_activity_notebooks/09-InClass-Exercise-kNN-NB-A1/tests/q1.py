from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 0

@test_case(points=None, hidden=False)
def test_scaling_shapes(X_train_s, X_test_s, X_train, X_test, acc_raw, acc_scaled):
    import numpy as np
    assert X_train_s.shape == X_train.shape and X_test_s.shape == X_test.shape
    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-06), 'scaled training columns should have mean 0'
    assert np.allclose(X_train_s.std(axis=0), 1, atol=1e-06), 'scaled training columns should have std 1'
    assert 0.0 <= acc_raw <= 1.0 and 0.0 <= acc_scaled <= 1.0
    assert acc_scaled > acc_raw, 'scaling should help k-NN on this data'

