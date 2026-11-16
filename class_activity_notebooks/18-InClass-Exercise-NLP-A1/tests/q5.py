from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 0

@test_case(points=None, hidden=False)
def test_cosine_and_nearest_toy(cosine_matrix, nearest):
    import numpy as np
    A = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    S = cosine_matrix(A)
    assert S.shape == (4, 4)
    assert np.allclose(np.diag(S), 1.0), 'a vector has cosine 1 with itself'
    assert np.allclose(S, S.T), 'cosine similarity is symmetric'
    assert abs(S[0, 1] - 1.0) < 1e-12, 'cosine ignores length'
    assert abs(S[0, 2]) < 1e-12, 'orthogonal vectors have cosine 0'
    assert abs(S[0, 3] - 1 / np.sqrt(2)) < 1e-12
    nn = nearest(S)
    assert nn.shape == (4,) and nn[0] == 1 and (nn[1] == 0), 'nearest must exclude the row itself'

@test_case(points=None, hidden=False)
def test_similarity_shapes(S_bow, S_tfidf, nn_bow, nn_tfidf, acc_bow, acc_tfidf, N):
    import numpy as np
    assert S_bow.shape == (N, N) and S_tfidf.shape == (N, N)
    assert np.all((S_bow >= -1e-09) & (S_bow <= 1 + 1e-09)) and np.all((S_tfidf >= -1e-09) & (S_tfidf <= 1 + 1e-09))
    assert len(nn_bow) == N and len(nn_tfidf) == N
    assert all((nn_bow[i] != i for i in range(N))) and all((nn_tfidf[i] != i for i in range(N)))
    assert isinstance(acc_bow, float) and isinstance(acc_tfidf, float)
    assert 0.0 <= acc_bow <= 1.0 and 0.0 <= acc_tfidf <= 1.0

