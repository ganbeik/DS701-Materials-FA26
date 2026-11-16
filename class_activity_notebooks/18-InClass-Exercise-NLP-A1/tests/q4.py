from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 0

@test_case(points=None, hidden=False)
def test_tfidf_sk_shape(idf_smooth, tfidf_sk, tfidf_ref, N, V):
    import numpy as np
    assert idf_smooth.shape == (V,) and tfidf_sk.shape == (N, V)
    assert np.all(idf_smooth >= 1.0), 'smoothed idf is at least 1'
    assert np.allclose(np.linalg.norm(tfidf_sk, axis=1), 1.0), 'rows must have unit L2 norm'
    assert np.allclose(tfidf_sk, tfidf_ref), 'hand-built matrix must match TfidfVectorizer'

@test_case(points=None, hidden=False)
def test_top_terms_shape(top_terms, vocab):
    out = top_terms(0, 3)
    assert isinstance(out, list) and len(out) == 3 and all((w in vocab for w in out))

