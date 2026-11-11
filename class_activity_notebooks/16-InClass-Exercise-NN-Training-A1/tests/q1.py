from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 0

@test_case(points=None, hidden=False)
def test_scaling(scaler, X_train_s, X_test_s, X_train, X_test):
    import numpy as np
    assert X_train_s.shape == X_train.shape and X_test_s.shape == X_test.shape
    assert np.allclose(X_train_s.mean(axis=0), 0, atol=1e-06), 'X_train_s must have zero column means (fit the scaler on X_train)'
    _varying = X_train.std(axis=0) > 0
    assert np.allclose(X_train_s.std(axis=0)[_varying], 1, atol=1e-06), 'X_train_s must have unit column std'
    assert np.allclose(X_test_s, scaler.transform(X_test)), 'X_test_s must be transformed with the scaler fitted on X_train (do not refit on the test set)'

@test_case(points=None, hidden=False)
def test_fit_and_curve(mlp, train_acc, test_acc, loss_curve):
    import numpy as np
    assert mlp.hidden_layer_sizes == (64,) and mlp.solver == 'adam'
    assert 0.9 <= test_acc <= 1.0, 'a properly scaled (64,) MLP should score above 0.9 on the test set'
    assert train_acc >= test_acc - 0.02
    assert len(loss_curve) == mlp.n_iter_, 'loss_curve should be mlp.loss_curve_ (one value per epoch)'
    assert loss_curve[-1] < 0.25 * loss_curve[0], 'the loss should have dropped substantially'
    assert np.mean(np.diff(loss_curve) < 0) >= 0.9, 'a healthy loss curve decreases in (almost) every epoch'

