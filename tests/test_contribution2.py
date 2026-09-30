"""Contribution 2 structural and leakage tests."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from federated.personalized_rf import PooledRandomForest, personalize_global_model

ROOT = __import__('pathlib').Path(__file__).resolve().parent.parent


def test_ten_clients_and_disjoint_assignments():
    for task in ['diagnosis','severity']:
        a=pd.read_csv(ROOT/f'results/contribution2/{task}/sample_assignments.csv')
        assert a.client_id.nunique()==10
        assert a.groupby('OASISID').client_id.nunique().max()==1
        assert a.groupby('OASISID').split.nunique().max()==1
        assert set(a.split)=={'federation','personalization','test'}


def test_client_splits_have_no_row_overlap():
    for task in ['diagnosis','severity']:
        a=pd.read_csv(ROOT/f'results/contribution2/{task}/sample_assignments.csv')
        for cid,g in a.groupby('client_id'):
            sets={s:set(g.loc[g.split==s,'OASISID']) for s in ['federation','personalization','test']}
            assert not (sets['federation'] & sets['personalization'])
            assert not (sets['federation'] & sets['test'])
            assert not (sets['personalization'] & sets['test'])


def test_global_has_100_trees_and_personalization_adds_trees():
    rng=np.random.RandomState(42)
    X=rng.randn(180,5); y=np.tile(np.arange(3),60)
    forests=[]
    for seed in [1,2]:
        forests.append(RandomForestClassifier(n_estimators=5,random_state=seed).fit(X,y))
    global_model=PooledRandomForest(forests,[0,1,2])
    personalized,local=personalize_global_model(global_model,X[:60],y[:60],n_trees=3,seed=9)
    assert len(global_model.estimators_)==10
    assert personalized.n_global_trees==10
    assert personalized.n_local_trees==3
    assert personalized.n_estimators==13
    assert personalized.global_estimators_ == global_model.estimators_
    assert not np.shares_memory(personalized.predict_proba(X[:5]), global_model.predict_proba(X[:5]))
    assert np.allclose(personalized.predict_proba(X[:5]).sum(axis=1),1.0)


def test_existing_artifacts_and_contribution2_outputs_exist():
    assert (ROOT/'results'/'shap_audit.csv').exists()
    assert (ROOT/'results'/'severity_metrics.csv').exists()
    for task in ['diagnosis','severity']:
        for name in ['client_distributions.csv','local_metrics.csv','global_metrics.csv','personalized_metrics.csv','personalization_gains.csv','summary.csv','shap_audit.csv']:
            assert (ROOT/'results'/'contribution2'/task/name).exists(), (task,name)
