from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 0

@test_case(points=None, hidden=False)
def test_cond_basics(cond_raw, cond_std, cond_no_copies):
    assert all((isinstance(v, float) for v in (cond_raw, cond_std, cond_no_copies)))
    assert cond_raw > cond_std, 'standardizing removes the part of kappa that is only units'
    assert cond_std > cond_no_copies >= 1, 'removing the near-duplicate columns should lower kappa'

