from otter.test_files import test_case

OK_FORMAT = False

name = "q3"
points = 0

@test_case(points=None, hidden=False)
def test_tfidf_shapes(df, idf, tfidf, n_zero_idf, bow, word_index, N, V):
    import numpy as np
    assert df.shape == (V,) and idf.shape == (V,) and (tfidf.shape == (N, V))
    assert df[word_index['the']] == N, "'the' is in every document"
    assert abs(idf[word_index['the']]) < 1e-12, 'a term in every document must have idf 0'
    assert np.all(idf >= 0), 'idf is never negative with the textbook formula'
    assert np.all(tfidf[:, word_index['the']] == 0)
    assert isinstance(n_zero_idf, int) and n_zero_idf >= 1

@test_case(points=None, hidden=False)
def test_top_term_callable(top_term, vocab, N):
    for i in range(N):
        assert top_term(i) in vocab

