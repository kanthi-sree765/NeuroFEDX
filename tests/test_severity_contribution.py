import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.severity_prediction import severity_from_cdr, TARGET_DERIVED_COLUMNS


def test_severity_mapping_exact():
    s = pd.Series([0, 0.5, 1, 2, 3, -1, np.nan])
    out = severity_from_cdr(s).astype('object').where(lambda x: x.notna(), None).tolist()
    assert out[:5] == [0, 1, 2, 2, 2]
    assert out[5] is None
    assert out[6] is None


def test_severity_artifacts_have_expected_distribution():
    p = ROOT / 'results' / 'severity_class_distribution.csv'
    df = pd.read_csv(p)
    assert df['severity_label'].tolist() == [0, 1, 2]
    assert df['n_rows'].tolist() == [4680, 1016, 340]


def test_severity_excluded_features_not_selected():
    selected = set(pd.read_csv(ROOT / 'results' / 'severity_selected_features.csv')['selected_feature'])
    assert selected.isdisjoint(TARGET_DERIVED_COLUMNS)
    assert 'CDRTOT' not in selected
    assert 'CDRSUM' not in selected


def test_severity_split_no_overlap():
    audit = pd.read_csv(ROOT / 'results' / 'severity_split_audit.csv')
    assert int(audit.loc[audit['set'] == 'overlap', 'n_subjects'].iloc[0]) == 0


def test_severity_probabilities_valid():
    pred = pd.read_csv(ROOT / 'results' / 'severity_predictions.csv')
    for prefix in ['severity_prob_c', 'severity_fed_prob_c']:
        cols = [f'{prefix}{i}' for i in range(3)]
        assert np.allclose(pred[cols].sum(axis=1), 1.0, atol=1e-8)
        assert (pred[cols].to_numpy() >= 0).all()
