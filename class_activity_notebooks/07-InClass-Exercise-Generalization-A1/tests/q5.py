from otter.test_files import test_case

OK_FORMAT = False

name = "q5"
points = 3

@test_case(points=None, hidden=False)
def test_mystery_shapes(mystery_gap, mystery_val_drop, diagnosis, more_data_helps, MYSTERY_TRAIN, MYSTERY_VAL, MYSTERY_N):
    import numpy as np
    assert np.allclose(mystery_gap, np.asarray(MYSTERY_VAL) - np.asarray(MYSTERY_TRAIN)), 'mystery_gap = validation - training, elementwise'
    assert isinstance(mystery_val_drop, float) and abs(mystery_val_drop - (MYSTERY_VAL[0] - MYSTERY_VAL[-1])) < 1e-09
    assert diagnosis in ('high bias', 'high variance'), 'diagnosis must be exactly "high bias" or "high variance"'
    assert isinstance(more_data_helps, bool), 'more_data_helps must be a Python bool'

