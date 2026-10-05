from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 4

@test_case(points=None, hidden=False)
def test_per_tree_predictions(per_tree_predictions, rf, X_test, y_test):
    import numpy as np
    P = per_tree_predictions(rf, X_test)
    assert P.shape == (300, len(X_test)), 'expected one row per tree, one column per test row'
    assert set(np.unique(P)) <= {0, 1}, 'rows must be hard 0/1 class predictions'
    vote = (P.mean(axis=0) > 0.5).astype(int)
    assert (vote == rf.predict(X_test)).mean() > 0.98

@test_case(points=None, hidden=False)
def test_mf_results(mf_results, MAX_FEATURES):
    import numpy as np, pandas as pd
    assert isinstance(mf_results, pd.DataFrame)
    assert list(mf_results.columns) == ['max_features', 'oob_score', 'test_acc', 'mean_tree_acc', 'mean_tree_corr']
    assert list(mf_results['max_features']) == list(MAX_FEATURES)
    first, last = (mf_results.iloc[0], mf_results.iloc[-1])
    assert first['mean_tree_acc'] < last['mean_tree_acc'], 'trees that see 1 feature per split should be individually weaker than trees that see all 30'
    assert first['mean_tree_corr'] < last['mean_tree_corr'], 'trees that see 1 feature per split should be LESS correlated than plain bagged trees'
    assert (mf_results['test_acc'] > mf_results['mean_tree_acc']).all()
    sqrt_row = mf_results[mf_results['max_features'] == 5].iloc[0]
    assert sqrt_row['oob_score'] >= last['oob_score'] - 1e-12

