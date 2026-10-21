from otter.test_files import test_case

OK_FORMAT = False

name = "q6"
points = 0

@test_case(points=None, hidden=False)
def test_l1_logistic_basics(n_kept_C, l1_logit, kept_l1, Cs):
    import numpy as np
    assert len(n_kept_C) == len(Cs)
    assert n_kept_C[0] <= n_kept_C[-1], 'larger C means less penalty, so more attributes survive'
    assert l1_logit.penalty == 'l1', 'the short list needs the L1 penalty'
    assert isinstance(kept_l1, list) and kept_l1 == sorted(kept_l1)

