from otter.test_files import test_case

OK_FORMAT = False

name = "q2"
points = 0

@test_case(points=None, hidden=False)
def test_bow_shape(bow, bow_sklearn, tokens, N, V):
    import numpy as np
    assert bow.shape == (N, V) and bow_sklearn.shape == (N, V)
    assert np.issubdtype(bow.dtype, np.integer), 'counts should be integers'
    assert (bow.sum(axis=1) == np.array([len(t) for t in tokens])).all(), "each row must sum to the document's token count"
    assert np.array_equal(bow, bow_sklearn), "hand-built bow must equal CountVectorizer's matrix"

