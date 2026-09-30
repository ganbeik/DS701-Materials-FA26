from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 3

@test_case(points=None, hidden=False)
def test_cv_function(cv_mse_for_degree):
    import numpy as np
    from sklearn.model_selection import KFold
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.0, 2.0, 2.0])
    out = cv_mse_for_degree(x, y, 0, KFold(n_splits=2, shuffle=False))
    assert isinstance(out, float), 'return a float (the mean over folds)'
    assert abs(out - 4.0) < 1e-09, f'expected 4.0 on the toy example, got {out}'

@test_case(points=None, hidden=False)
def test_cv_choice_shape(cv_mse, cv_degree, test_mse_at_cv_degree, test_mse, DEGREES):
    import numpy as np
    cv_mse = np.asarray(cv_mse, dtype=float)
    assert cv_mse.shape == (len(DEGREES),), 'cv_mse needs one entry per degree'
    assert np.all(cv_mse >= 0)
    assert isinstance(cv_degree, (int, np.integer)) and cv_degree == int(DEGREES[np.argmin(cv_mse)]), 'cv_degree must be the argmin of cv_mse'
    assert isinstance(test_mse_at_cv_degree, float) and abs(test_mse_at_cv_degree - float(test_mse[cv_degree])) < 1e-12
    assert cv_mse[-1] > 10 * cv_mse[cv_degree], 'with 12-13 training points per fold, degree 12 should be a CV disaster'

