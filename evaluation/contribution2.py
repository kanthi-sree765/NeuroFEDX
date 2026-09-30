"""Contribution 2: Personalized Federated Learning using Client-Specific Local Adaptation.

Runs diagnosis first and then the same personalization mechanism for CDR-based
severity. The frozen base-paper experiment is never modified; this module uses
its existing RF tree-pooling primitives on a separate controlled non-IID
experiment.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, f1_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR, KNN_NEIGHBORS, PEARSON_THRESHOLD, SMOTE_RANDOM_STATE
from preprocessing.preprocess_common import pearson_prune, smote_resample
from models.random_forest import train_rf
from federated.federated_rf import train_federated_round
from federated.personalized_rf import personalize_global_model, train_non_iid_federated
from explainability.contribution2_shap import run_task as run_c2_shap

SEED = 42
N_CLIENTS = 10
DIRICHLET_ALPHA = 1.0
MIN_CLIENT_SAMPLES = 50
FEDERATION_PROP = 0.60
PERSONALIZATION_PROP = 0.20
TEST_PROP = 0.20
PERSONALIZATION_TREES = 10


def participant_train_test(df, seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr, te = next(gss.split(df, groups=df.OASISID))
    return np.asarray(tr), np.asarray(te)


def subject_labels(df, label_col):
    rows = []
    for sid, g in df.groupby("OASISID", sort=True):
        counts = g[label_col].value_counts()
        label = counts.index[0]
        rows.append((sid, label, len(g)))
    return pd.DataFrame(rows, columns=["OASISID", label_col, "n_rows"])


def dirichlet_subject_assignment(df, label_col, n_clients=N_CLIENTS,
                                 alpha=DIRICHLET_ALPHA, seed=SEED):
    """Assign whole participants to clients using one fixed Dirichlet allocation per class."""
    subjects = subject_labels(df, label_col)
    rng = np.random.RandomState(seed)
    assignment = {}
    for class_value in sorted(subjects[label_col].unique(), key=str):
        sids = subjects.loc[subjects[label_col] == class_value, "OASISID"].to_numpy().copy()
        rng.shuffle(sids)
        proportions = rng.dirichlet(np.full(n_clients, alpha, dtype=float))
        counts = rng.multinomial(len(sids), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            for sid in sids[start:start + count]:
                assignment[sid] = client_id
            start += count
    out = df.copy()
    out["client_id"] = out.OASISID.map(assignment).astype(int)
    return out, subjects


def split_client_subjects(df, seed=SEED):
    """Within each client, split whole subjects into federation/personalization/test."""
    rng = np.random.RandomState(seed)
    rows = []
    for client_id, cdf in df.groupby("client_id", sort=True):
        sids = cdf.OASISID.drop_duplicates().to_numpy().copy()
        rng.shuffle(sids)
        n = len(sids)
        n_test = max(1, int(round(n * TEST_PROP)))
        n_personal = max(1, int(round(n * PERSONALIZATION_PROP)))
        if n_test + n_personal >= n:
            n_personal = max(1, n - n_test - 1)
        test_sids = set(sids[:n_test])
        personal_sids = set(sids[n_test:n_test + n_personal])
        fed_sids = set(sids[n_test + n_personal:])
        for sid in sids:
            if sid in test_sids: split = "test"
            elif sid in personal_sids: split = "personalization"
            elif sid in fed_sids: split = "federation"
            else: raise AssertionError("Unassigned subject")
            rows.append((sid, int(client_id), split))
    return pd.DataFrame(rows, columns=["OASISID", "client_id", "split"])


def audit_assignments(df, assignments, label_col, out_dir):
    merged = df[["OASISID", label_col]].merge(assignments, on="OASISID", how="left")
    assert merged["split"].notna().all()
    assert merged.groupby("OASISID")["client_id"].nunique().max() == 1
    assert merged.groupby("OASISID")["split"].nunique().max() == 1
    # sample-level audit: each row has exactly one assignment
    assert len(merged) == len(df)
    drows=[]
    for (client, split), g in merged.groupby(["client_id","split"], sort=True):
        row={"client_id":client,"split":split,"n_rows":len(g),"n_subjects":g.OASISID.nunique()}
        for c,n in g[label_col].value_counts().items(): row[str(c)]=int(n)
        drows.append(row)
    dist=pd.DataFrame(drows).fillna(0)
    dist.to_csv(out_dir/"client_distributions.csv", index=False)
    assignments.to_csv(out_dir/"sample_assignments.csv", index=False)
    return merged, dist


def preprocess_for_c2(df, label_col, feature_cols, assignments=None):
    """Fit imputation/Pearson on all training-side rows; transform client tests untouched."""
    if assignments is not None:
        base = df.drop(columns=[c for c in ["client_id", "split"] if c in df.columns], errors="ignore")
        df = base.merge(assignments, on="OASISID", how="left", validate="many_to_one")
    train_mask = df["split"].isin(["federation", "personalization"])
    test_mask = df["split"].eq("test")
    train_rows = df.loc[train_mask].copy()
    test_rows = df.loc[test_mask].copy()
    imputer=KNNImputer(n_neighbors=KNN_NEIGHBORS)
    Xtr_imp=imputer.fit_transform(train_rows[feature_cols])
    Xte_imp=imputer.transform(test_rows[feature_cols])
    tr_imp=pd.DataFrame(Xtr_imp,columns=feature_cols,index=train_rows.index)
    te_imp=pd.DataFrame(Xte_imp,columns=feature_cols,index=test_rows.index)
    selected,dropped=pearson_prune(tr_imp, feature_cols, threshold=PEARSON_THRESHOLD, verbose=False)
    out_train=train_rows[["OASISID",label_col,"client_id","split"]].copy()
    out_test=test_rows[["OASISID",label_col,"client_id","split"]].copy()
    for c in selected:
        out_train[c]=tr_imp[c]
        out_test[c]=te_imp[c]
    return out_train.reset_index(drop=True), out_test.reset_index(drop=True), selected, dropped

def metric_dict(y_true, y_pred, proba, labels, model, client_id):
    p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
    try:
        auc=roc_auc_score(label_binarize(y_true,classes=labels),proba,average="macro",multi_class="ovr",labels=None)
    except Exception:
        auc=np.nan
    return {"client_id":client_id,"model":model,"accuracy":accuracy_score(y_true,y_pred),
            "balanced_accuracy":balanced_accuracy_score(y_true,y_pred),"macro_precision":p.mean(),
            "macro_recall":r.mean(),"macro_f1":f1_score(y_true,y_pred,labels=labels,average="macro",zero_division=0),
            "weighted_f1":f1_score(y_true,y_pred,labels=labels,average="weighted",zero_division=0),
            "auc_macro_ovr":auc,"n_test":len(y_true)}


def evaluate_client(model, X, y, labels, name, client_id, out_dir):
    pred=model.predict(X); proba=model.predict_proba(X)
    row=metric_dict(y,pred,proba,labels,name,client_id)
    p,r,f,s=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    per=pd.DataFrame({"client_id":client_id,"model":name,"class":labels,"precision":p,"recall":r,"f1":f,"support":s})
    cm=confusion_matrix(y,pred,labels=labels)
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(out_dir/"confusion_matrices"/f"client_{client_id:02d}_{name}.csv")
    return row,per


def summarize_metrics(df):
    metrics=[c for c in ["accuracy","balanced_accuracy","macro_precision","macro_recall","macro_f1","weighted_f1","auc_macro_ovr"] if c in df]
    rows=[]
    for model,g in df.groupby("model"):
        for metric in metrics:
            vals=g[metric].dropna()
            rows.append({"model":model,"metric":metric,"mean":vals.mean(),"std":vals.std(ddof=1),"min":vals.min(),"max":vals.max()})
    return pd.DataFrame(rows)


def run_task(df, label_col, feature_cols, task_name, out_root):
    out_dir=out_root/task_name; (out_dir/"confusion_matrices").mkdir(parents=True,exist_ok=True)
    tr_idx, outer_test_idx=participant_train_test(df)
    work=df.iloc[tr_idx].reset_index(drop=True).copy()
    outer_test=df.iloc[outer_test_idx].reset_index(drop=True).copy()
    outer_overlap=set(work.OASISID) & set(outer_test.OASISID)
    pd.DataFrame([{
        "outer_train_subjects":work.OASISID.nunique(),
        "outer_test_subjects":outer_test.OASISID.nunique(),
        "outer_subject_overlap":len(outer_overlap),
        "client_experiment_subjects":work.OASISID.nunique(),
        "client_experiment_uses_outer_test":bool(set(outer_test.OASISID) & set(work.OASISID))
    }]).to_csv(out_dir/"outer_split_audit.csv",index=False)
    assert not outer_overlap
    assigned, _=dirichlet_subject_assignment(work,label_col,seed=SEED)
    client_row_counts = assigned.groupby("client_id").size()
    if len(client_row_counts) != N_CLIENTS or int(client_row_counts.min()) < MIN_CLIENT_SAMPLES:
        raise RuntimeError(
            f"Dirichlet allocation violated minimum client size {MIN_CLIENT_SAMPLES}: "
            f"{client_row_counts.to_dict()}"
        )
    subject_client=assigned[["OASISID","client_id"]].drop_duplicates("OASISID")
    split_map=split_client_subjects(subject_client,seed=SEED)
    assigned=split_map[["OASISID","client_id","split"]].copy()
    work2=work.merge(assigned,on="OASISID",how="left",validate="many_to_one")
    audit_assignments(work2,assigned,label_col,out_dir)

    train_proc,test_proc,selected,dropped=preprocess_for_c2(work2,label_col,feature_cols,assigned)
    pd.DataFrame({"selected_feature":selected}).to_csv(out_dir/"selected_features.csv",index=False)
    pd.DataFrame({"dropped_feature":dropped}).to_csv(out_dir/"pearson_dropped_features.csv",index=False)
    # All models use only the controlled client-held-out test rows, not the original outer test.
    # The outer test remains untouched and is used only as a secondary leakage audit artifact.
    client_models={}; global_model=None
    fed_shards=[]
    local_rows=[]; global_rows=[]; pers_rows=[]; per_rows=[]; tree_audit=[]
    client_ids=sorted(train_proc.client_id.unique())
    assert len(client_ids)==N_CLIENTS
    labels=np.sort(train_proc[label_col].unique())
    for cid in client_ids:
        ctrain=train_proc[(train_proc.client_id==cid)&(train_proc.split=="federation")]
        cpers=train_proc[(train_proc.client_id==cid)&(train_proc.split=="personalization")]
        ctest=test_proc[test_proc.client_id==cid]
        Xf,yf=smote_resample(ctrain[selected].to_numpy(),ctrain[label_col].to_numpy(),k_neighbors=5,random_state=SMOTE_RANDOM_STATE)
        # Personalization remains private; apply SMOTE only to this training-side set.
        Xp,yp=smote_resample(cpers[selected].to_numpy(),cpers[label_col].to_numpy(),k_neighbors=5,random_state=SMOTE_RANDOM_STATE)
        fed_shards.append((Xf,yf))
        client_models[cid]=(Xf,yf,Xp,yp,ctest)

    # 50 rounds using the same RF-specific tree pooling as the base method, but on C2's non-IID federation shards.
    global_model, global_forests, global_history = train_non_iid_federated(
        fed_shards, n_rounds=50, total_trees=100, seed=SEED, classes=labels
    )
    pd.DataFrame(global_history).to_csv(out_dir/"global_round_history.csv",index=False)

    for cid in client_ids:
        Xf,yf,Xp,yp,ctest=client_models[cid]
        Xtest=ctest[selected].to_numpy(); ytest=ctest[label_col].to_numpy()
        # Local-only diagnosis/severity model uses federation-training data only.
        local_model=train_rf(Xf,yf)
        # Personalized model = exact global forest + additional local personalization trees.
        personalized, local_personal_model = __import__('federated.personalized_rf',fromlist=['personalize_global_model']).personalize_global_model(
            global_model,Xp,yp,n_trees=PERSONALIZATION_TREES,seed=SEED+cid)
        r,pc=evaluate_client(local_model,Xtest,ytest,labels,"local",cid,out_dir); local_rows.append(r); per_rows.append(pc)
        r,pc=evaluate_client(global_model,Xtest,ytest,labels,"global",cid,out_dir); global_rows.append(r); per_rows.append(pc)
        r,pc=evaluate_client(personalized,Xtest,ytest,labels,"personalized",cid,out_dir); pers_rows.append(r); per_rows.append(pc)
        client_models[cid]=(*client_models[cid],local_model,personalized,local_personal_model)
        tree_audit.append({
            "client_id":cid, "global_tree_count":personalized.n_global_trees,
            "personalization_tree_count":personalized.n_local_trees,
            "personalized_tree_count":personalized.n_estimators,
            "personalization_source":"client_personalization_train_only"
        })

    local_df=pd.DataFrame(local_rows); global_df=pd.DataFrame(global_rows); pers_df=pd.DataFrame(pers_rows)
    local_df.to_csv(out_dir/"local_metrics.csv",index=False)
    global_df.to_csv(out_dir/"global_metrics.csv",index=False)
    pers_df.to_csv(out_dir/"personalized_metrics.csv",index=False)
    all_df=pd.concat([local_df,global_df,pers_df],ignore_index=True)
    summarize_metrics(all_df).to_csv(out_dir/"summary.csv",index=False)
    pd.concat(per_rows,ignore_index=True).to_csv(out_dir/"per_class_metrics.csv",index=False)
    pd.DataFrame(tree_audit).to_csv(out_dir/"personalization_tree_audit.csv",index=False)
    # Save the global model once; personalized client models are reproducibly rebuilt by this script.
    import joblib
    joblib.dump({"global_model": global_model, "selected_features": selected, "labels": labels},
                out_dir/"global_model.joblib")
    gains=pers_df.merge(global_df,on="client_id",suffixes=("_personalized","_global"))
    for m in ["accuracy","balanced_accuracy","macro_f1","auc_macro_ovr"]:
        gains[f"personalization_gain_{m}"]=gains[f"{m}_personalized"]-gains[f"{m}_global"]
    gains[["client_id"]+[c for c in gains.columns if c.startswith("personalization_gain_")]].to_csv(out_dir/"personalization_gains.csv",index=False)
    # Model metadata/audit.
    # Actual TreeSHAP for the global tree pool and each personalized union.
    run_c2_shap(global_forests, client_models, selected, labels, task_name, out_dir, per_client_samples=5)

    metadata={"task":task_name,"n_clients":10,"rounds":50,"global_trees":100,"personalization_trees":10,
              "dirichlet_alpha":DIRICHLET_ALPHA,"seed":SEED,"federation_proportion":FEDERATION_PROP,
              "personalization_proportion":PERSONALIZATION_PROP,"test_proportion":TEST_PROP,
              "min_client_samples":MIN_CLIENT_SAMPLES,"client_assignment_seed":SEED,
              "rf_params":__import__('config.model_config',fromlist=['RF_PARAMS']).RF_PARAMS,"selected_features":selected}
    (out_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return all_df, selected, work2, client_models, global_model


def main():
    import sys
    root=RESULTS_DIR/"contribution2"; root.mkdir(exist_ok=True)
    requested = sys.argv[1:] or ["diagnosis", "severity"]
    results = {}
    if "diagnosis" in requested:
        base=pd.read_csv(RESULTS_DIR/"base_dataframe_unimputed.csv")
        diagnosis_features=[c for c in base.columns if c not in ("OASISID","diagnosis_class","MRI_matched")]
        results["diagnosis"] = run_task(base,"diagnosis_class",diagnosis_features,"diagnosis",root)[0]
    if "severity" in requested:
        sev=pd.read_csv(RESULTS_DIR/"severity_labelled_dataset.csv")
        sev_features=[c for c in sev.columns if c not in ("OASISID","severity_label","CDRTOT_source")]
        results["severity"] = run_task(sev,"severity_label",sev_features,"severity",root)[0]
    print("Contribution 2 complete")
    for task,df in results.items():
        print(f"{task.title()} summary:\n",df.groupby("model")["accuracy"].agg(["mean","std","min","max"]))

if __name__=='__main__': main()
