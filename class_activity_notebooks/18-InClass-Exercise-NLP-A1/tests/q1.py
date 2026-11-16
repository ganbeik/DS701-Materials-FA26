from otter.test_files import test_case

OK_FORMAT = False

name = "q1"
points = 0

@test_case(points=None, hidden=False)
def test_tokenize_basic(tokenize):
    assert tokenize('The cat sat on the mat.') == ['the', 'cat', 'sat', 'on', 'the', 'mat']
    assert tokenize('Show me the money!') == ['show', 'me', 'the', 'money']
    assert tokenize('A 3D-printed part') == ['printed', 'part'], 'single letters and digits are not tokens under TOKEN_PATTERN'

@test_case(points=None, hidden=False)
def test_vocab_shape(tokens, vocab, V, docs):
    assert len(tokens) == len(docs) and all((isinstance(t, list) for t in tokens))
    assert vocab == sorted(vocab), 'vocab must be sorted'
    assert len(set(vocab)) == len(vocab), 'vocab must not contain duplicates'
    assert all((w == w.lower() for w in vocab)), 'tokens must be lowercased'
    assert isinstance(V, int) and V == len(vocab)

