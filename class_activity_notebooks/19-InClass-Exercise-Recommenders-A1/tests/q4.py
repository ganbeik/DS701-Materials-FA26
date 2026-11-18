from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 4

@test_case(points=None, hidden=False)
def test_als_shape(ridge_rows, U, V, obj_hist, rmse_hist, pred_mf, rmse_mf, K_MF, LAM, n_users, n_items, holdout):
    import numpy as np
    assert K_MF == 4 and LAM == 1.0
    assert U.shape == (n_users, K_MF) and V.shape == (n_items, K_MF), 'U is users x k, V is movies x k'
    assert len(obj_hist) == len(rmse_hist) >= 3
    assert all((b <= a + 1e-06 for a, b in zip(obj_hist, obj_hist[1:]))), 'the ALS objective must never increase'
    assert obj_hist[-2] - obj_hist[-1] < 0.0001 or len(obj_hist) == 50, 'run until the objective stops improving'
    assert rmse_hist[-1] < rmse_hist[0]
    assert np.asarray(pred_mf).shape == (len(holdout),)
    F = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Rc = np.array([[1.0, 2.0, 3.0]])
    M = np.ones((1, 3), bool)
    ref = np.linalg.solve(F.T @ F + 0.5 * np.eye(2), F.T @ Rc[0])
    assert np.allclose(ridge_rows(Rc, M, F, 0.5), ref[None, :])
    assert np.allclose(ridge_rows(Rc, np.zeros((1, 3), bool), F, 0.5), 0.0), 'a row with nothing observed gets zeros'

