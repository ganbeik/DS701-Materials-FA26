from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 4

@test_case(points=None, hidden=False)
def test_depth_results_shape(depth_results, DEPTHS):
    import pandas as pd
    assert isinstance(depth_results, pd.DataFrame), 'depth_results must be a DataFrame'
    assert list(depth_results.columns[:3]) == ['max_depth', 'train_acc', 'test_acc']
    assert len(depth_results) == len(DEPTHS)
    assert list(depth_results['max_depth']) == list(DEPTHS)

@test_case(points=None, hidden=False)
def test_overfitting_story(depth_results, best_tree_test_acc, full_tree, X_train, y_train, X_test, y_test):
    import numpy as np
    assert (depth_results['train_acc'].diff().dropna() >= -1e-12).all(), 'training accuracy must be non-decreasing in max_depth'
    assert np.isclose(depth_results['train_acc'].iloc[-1], 1.0)
    assert np.isclose(full_tree.score(X_train, y_train), 1.0)
    assert best_tree_test_acc > full_tree.score(X_test, y_test), 'the pruned tree should beat the fully grown tree on the test set'
    gap = depth_results['train_acc'] - depth_results['test_acc']
    assert gap.iloc[-1] > gap.iloc[0]

