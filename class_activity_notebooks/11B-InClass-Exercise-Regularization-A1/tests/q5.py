from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 0

@test_case(points=None, hidden=False)
def test_raw_basics(selected_raw, mse_raw, mse_lasso, dropped_true_raw, TRUE_IDX):
    assert isinstance(selected_raw, list) and isinstance(dropped_true_raw, list)
    assert set(dropped_true_raw) <= set(TRUE_IDX)
    assert mse_raw > mse_lasso, 'the unstandardized fit should predict worse at the same alpha'

