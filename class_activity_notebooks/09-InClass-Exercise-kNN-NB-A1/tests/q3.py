from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 0

@test_case(points=None, hidden=False)
def test_bakeoff_shape(models, cms, results, n_test):
    import numpy as np
    assert set(models) == set(cms) == set(results) == {'knn', 'nb', 'tree'}
    for name in models:
        cm = np.asarray(cms[name])
        assert cm.shape == (2, 2) and cm.sum() == n_test, f'{name}: confusion matrix must be 2x2 over all test points'
        tn, fp, fn, tp = cm.ravel()
        r = results[name]
        assert set(r) == {'accuracy', 'precision', 'recall'}
        assert abs(r['precision'] - tp / (tp + fp)) < 1e-09, f'{name}: precision must be TP/(TP+FP)'
        assert abs(r['recall'] - tp / (tp + fn)) < 1e-09, f'{name}: recall must be TP/(TP+FN)'
        assert abs(r['accuracy'] - (tp + tn) / cm.sum()) < 1e-09

