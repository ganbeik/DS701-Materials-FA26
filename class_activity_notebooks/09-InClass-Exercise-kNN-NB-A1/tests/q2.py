from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 0

@test_case(points=None, hidden=False)
def test_ksweep_shape(cv_acc, best_metric, best_k, acc_knn_test, K_VALUES, METRICS):
    import numpy as np
    assert set(cv_acc) == set(METRICS), 'cv_acc needs exactly the two METRICS as keys'
    for m in METRICS:
        a = np.asarray(cv_acc[m])
        assert a.shape == (len(K_VALUES),), f'{m}: one mean CV accuracy per k in K_VALUES'
        assert np.all((a >= 0) & (a <= 1))
    assert best_metric in METRICS and 1 <= best_k <= 25
    top = max((np.max(cv_acc[m]) for m in METRICS))
    assert abs(cv_acc[best_metric][best_k - 1] - top) < 1e-12, 'best_k/best_metric must achieve the highest CV accuracy'
    assert 0.0 <= acc_knn_test <= 1.0

