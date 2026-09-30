from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 3

@test_case(points=None, hidden=False)
def test_curve_shapes(train_mse, test_mse, best_degree, DEGREES):
    import numpy as np
    train_mse, test_mse = (np.asarray(train_mse, dtype=float), np.asarray(test_mse, dtype=float))
    assert train_mse.shape == (len(DEGREES),), 'train_mse needs one entry per degree'
    assert test_mse.shape == (len(DEGREES),), 'test_mse needs one entry per degree'
    assert np.all(train_mse >= 0) and np.all(test_mse >= 0)
    assert np.all(np.diff(train_mse) <= 1e-09), 'training MSE can never go UP when the degree increases (a bigger model can always reproduce a smaller one)'
    assert isinstance(best_degree, (int, np.integer)) and int(DEGREES[np.argmin(test_mse)]) == best_degree, 'best_degree must be the argmin of test_mse'
    assert test_mse[-1] > test_mse[best_degree], 'test error should be worse at degree 12 than at the best degree -- overfitting'

