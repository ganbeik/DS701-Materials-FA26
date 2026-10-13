from otter.test_files import test_case

OK_FORMAT = False

name = "q4"
points = 0

@test_case(points=None, hidden=False)
def test_nb_by_hand_lecture_table(nb_scores, scores_raw, scores_smooth, nb_prediction, p_spam_smooth):
    import pandas as pd
    tax = pd.DataFrame({'Refund': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No'], 'Marital': ['Single', 'Married', 'Single', 'Married', 'Divorced', 'Married', 'Divorced', 'Single', 'Married', 'Single'], 'Evade': ['No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes']})
    s = nb_scores(tax, {'Refund': 'No', 'Marital': 'Married'}, 'Evade', alpha=0)
    assert set(s) == {'No', 'Yes'}
    assert abs(s['No'] - 0.7 * 4 / 7 * 4 / 7) < 1e-09, 'Evade=No should score 0.7 * 4/7 * 4/7'
    assert abs(s['Yes']) < 1e-12, 'Evade=Yes should score exactly 0 (no married evader in the table)'
    assert set(scores_raw) == set(scores_smooth) == {'yes', 'no'}
    assert min(scores_raw.values()) == 0.0, "your section's query should hit an empty count without smoothing"
    assert min(scores_smooth.values()) > 0.0, 'smoothing should remove the zero'
    assert nb_prediction in ('yes', 'no') and 0.0 <= p_spam_smooth <= 1.0

