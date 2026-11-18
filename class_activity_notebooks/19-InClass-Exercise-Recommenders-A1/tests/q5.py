from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 2

@test_case(points=None, hidden=False)
def test_latent_shape(V_pca, nn_genre_purity, n_items, n_genres):
    import numpy as np
    assert np.asarray(V_pca).shape == (n_items, 2)
    assert isinstance(nn_genre_purity, float) and 0.0 <= nn_genre_purity <= 1.0
    assert nn_genre_purity > 1 / n_genres, 'purity should be well above chance'

