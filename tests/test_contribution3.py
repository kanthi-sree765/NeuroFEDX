import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
RES=ROOT/'results'/'contribution3'


def test_outputs_exist_and_nonempty():
    for task in ['diagnosis','severity']:
        for f in ['patient_level_explanations.csv','modality_contributions.csv','top_features.csv','global_vs_personalized_explanations.csv','explanation_similarity.csv','client_modality_summary.csv']:
            p=RES/task/f; assert p.exists() and p.stat().st_size>0


def test_modality_mapping_and_normalization():
    m=pd.read_csv(RES/'summary'/'feature_modality_mapping_audit.csv')
    assert set(m.modality).issubset({'Clinical / Demographic','Psychological / Cognitive','MRI','UNMAPPED'})
    assert not m.empty
    d=pd.read_csv(RES/'diagnosis'/'modality_contributions.csv')
    s=pd.read_csv(RES/'severity'/'modality_contributions.csv')
    for x in [d,s]:
        for _,g in x[x.modality!='UNMAPPED'].groupby(['sample_id','client_id','task','model_type']):
            total=g.normalized_contribution.sum()
            assert np.isclose(total,1.0,atol=1e-8)


def test_feature_rows_are_real_and_predicted_class_explained():
    for task in ['diagnosis','severity']:
        p=pd.read_csv(RES/task/'patient_level_explanations.csv')
        assert set(p.model_type)=={'global','personalized'}
        assert (p.feature!='__SUMMARY__').all()
        assert (p.predicted_class==p.explained_class).all()
        assert p.feature_value.notna().all()


def test_additivity_and_comparison():
    a=pd.read_csv(RES/'summary'/'shap_audit.csv')
    assert not a.empty
    assert a.max_additivity_error.max() <= 1e-8
    s=pd.read_csv(RES/'summary'/'explanation_similarity_summary.csv')
    assert not s.empty
    assert s.top5_overlap.between(0,1).all()
    assert s.top10_overlap.between(0,1).all()
