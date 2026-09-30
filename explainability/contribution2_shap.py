"""TreeSHAP for Contribution 2 tree-union global/personalized models.

Because non-IID clients can lack rare classes, the C2 global model is an
explicit class-aligned tree pool rather than a sklearn RF whose trees all
share identical class arrays. SHAP is therefore composed from actual
TreeExplainer results for each fitted RF component, weighted by its number of
trees. This is still TreeSHAP; no permutation-importance substitute is used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from explainability.shap_analysis import normalize_shap_values


def _align(values, expected, local_classes, global_classes, n_samples, n_features):
    vals = normalize_shap_values(values, n_samples, n_features, len(local_classes))
    out = [np.zeros((n_samples,n_features), dtype=float) for _ in global_classes]
    exp = np.zeros(len(global_classes), dtype=float)
    for j,c in enumerate(np.asarray(local_classes)):
        k = np.flatnonzero(np.asarray(global_classes)==c)
        if len(k):
            out[int(k[0])] = vals[j]
            exp[int(k[0])] = np.asarray(expected)[j]
    return out, exp


def explain_tree_union(component_models, X, global_classes):
    """Return TreeSHAP values/expected values for a union of RF components."""
    X=np.asarray(X); n=len(X); p=X.shape[1]; C=len(global_classes)
    total_trees=sum(len(m.estimators_) for m in component_models)
    values=[np.zeros((n,p),dtype=float) for _ in range(C)]
    expected=np.zeros(C,dtype=float)
    for model in component_models:
        if len(model.estimators_)==0: continue
        explainer=shap.TreeExplainer(model)
        raw=explainer.shap_values(X,check_additivity=True)
        aligned, exp=_align(raw,explainer.expected_value,model.classes_,global_classes,n,p)
        weight=len(model.estimators_)/total_trees
        for c in range(C):
            values[c] += aligned[c]*weight
            expected[c] += exp[c]*weight
    return values, expected


def audit_union(model, X, values, expected):
    proba=model.predict_proba(X)
    errs=[]
    for c in range(len(model.classes_)):
        errs.append(float(np.max(np.abs(expected[c]+values[c].sum(axis=1)-proba[:,c]))))
    return max(errs)


def run_task(global_forests, client_models, feature_names, labels, task_name, out_dir, per_client_samples=5):
    out_dir=Path(out_dir); rows=[]; importance_rows=[]
    for cid in sorted(client_models):
        data=client_models[cid]
        ctest=data[4]; personalized=data[-2]; local_personal=data[-1]
        cols=list(feature_names)
        X=ctest[cols].to_numpy()
        if len(X)==0: continue
        rng=np.random.RandomState(42+cid)
        picks=rng.choice(len(X),size=min(per_client_samples,len(X)),replace=False)
        Xs=X[picks]
        # Global tree union is identical for every client.
        gv,ge=explain_tree_union(global_forests,Xs,labels)
        gerr=audit_union(personalized.global_model,Xs,gv,ge)
        pv,pe=explain_tree_union(global_forests+[local_personal],Xs,labels)
        perr=audit_union(personalized,Xs,pv,pe)
        rows.extend([
            {"task":task_name,"client_id":cid,"model":"global","n_samples":len(Xs),"n_global_trees":100,"n_local_trees":0,"max_additivity_error":gerr,"shap_package_version":shap.__version__,"method":"TreeSHAP component composition"},
            {"task":task_name,"client_id":cid,"model":"personalized","n_samples":len(Xs),"n_global_trees":100,"n_local_trees":len(local_personal.estimators_),"max_additivity_error":perr,"shap_package_version":shap.__version__,"method":"TreeSHAP component composition"},
        ])
        for model_name, vals in [("global",gv),("personalized",pv)]:
            imp=np.vstack([np.mean(np.abs(v),axis=0) for v in vals]).mean(axis=0)
            for f,v in zip(cols,imp): importance_rows.append({"task":task_name,"client_id":cid,"model":model_name,"feature":f,"mean_abs_shap":v})
    pd.DataFrame(rows).to_csv(out_dir/"shap_audit.csv",index=False)
    imp=pd.DataFrame(importance_rows)
    imp.to_csv(out_dir/"shap_importance.csv",index=False)
    return pd.DataFrame(rows), imp
