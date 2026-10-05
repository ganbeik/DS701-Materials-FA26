from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 3

@test_case(points=None, hidden=False)
def test_gini(gini):
    import numpy as np
    assert gini is not None, 'define gini(counts) before running this check'
    assert np.isclose(gini([6, 0]), 0.0), 'a pure node has Gini 0'
    assert np.isclose(gini([3, 3]), 0.5), 'a 50/50 binary node has Gini 0.5'
    assert np.isclose(gini([0, 0]), 0.0), 'an empty node must not divide by zero'

@test_case(points=None, hidden=False)
def test_weighted_gini(gini, weighted_gini):
    import numpy as np
    assert weighted_gini is not None, 'define weighted_gini(left_counts, right_counts)'
    assert np.isclose(weighted_gini([5, 0], [0, 7]), 0.0), 'a perfect split drives collective impurity to zero'
    assert np.isclose(weighted_gini([2, 2], [2, 2]), gini([4, 4])), 'a split that separates nothing cannot change impurity'

