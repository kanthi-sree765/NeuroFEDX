"""Contribution 3: patient-level multimodal explainability for C2 models.

This module is post-hoc only: it loads already-trained C2 global forests, rebuilds
client personalization trees from the recorded private personalization split, and
uses actual TreeSHAP on those fitted trees. No model is retrained for explanation
selection or evaluation.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.model_config import RESULTS_DIR, KNN_NEIGHBORS, PEARSON_THRESHOLD, RF_PARAMS, SMOTE_RANDOM_STATE
from preprocessing.preprocess_common import pearson_prune, smote_resample
from federated.personalized_rf import PersonalizedRandomForest

SEED = 42
TOP_K = 5
SAMPLES_PER_CLIENT = 1
EXPLAIN_CLIENTS = [0, 5]
SHAP_ADD_TOL = 1e-8
MODALITY_ORDER = ["Clinical / Demographic", "Psychological / Cognitive", "MRI", "UNMAPPED"]


def load_mapping():
    p = RESULTS_DIR / "final_feature_mapping.csv"
    m = pd.read_csv(p)
    # Existing project terminology is preserved. The mapping's status is retained
    # for audit; interpreted mappings are not silently upgraded to verified.
    return m.set_index("paper_feature")


def modality_map_for(features):
    m = load_mapping()
    out = {}
    status = {}
    for f in features:
        if f in m.index:
            out[f] = {"Clinical": "Clinical / Demographic",
                      "Psychological": "Psychological / Cognitive",
                      "MRI": "MRI"}.get(str(m.loc[f, "modality"]), "UNMAPPED")
            status[f] = str(m.loc[f, "status"])
        else:
            out[f] = "UNMAPPED"; status[f] = "MISSING_FROM_MAPPING"
    return out, status


def _tree_shap_one(tree, X, global_classes):
    expl = shap.TreeExplainer(tree)
    vals = expl.shap_values(X, check_additivity=True)
    n, p = X.shape
    local_classes = np.asarray(getattr(tree, "classes_", []))
    # normalize the several SHAP output forms across versions
    if isinstance(vals, list):
        arrs = [np.asarray(v, dtype=float) for v in vals]
    else:
        a = np.asarray(vals, dtype=float)
        if a.ndim == 3:
            # newer multiclass TreeSHAP: samples x features x classes
            arrs = [a[:, :, c] for c in range(a.shape[2])]
        elif a.ndim == 2 and len(local_classes) == 2:
            # binary TreeSHAP may return one matrix; treat it as the positive class.
            arrs = [np.zeros((n, p), dtype=float), a]
        else:
            arrs = [a]
    expected = np.asarray(expl.expected_value, dtype=float).reshape(-1)
    if len(expected) == 1 and len(local_classes) == 2 and len(arrs) == 2:
        expected = np.array([1.0 - expected[0], expected[0]])
    out = [np.zeros((n, p), dtype=float) for _ in global_classes]
    ev = np.zeros(len(global_classes), dtype=float)
    for j, c in enumerate(local_classes):
        k = np.flatnonzero(np.asarray(global_classes) == c)
        if len(k) and j < len(arrs):
            out[int(k[0])] = arrs[j]
            if j < len(expected): ev[int(k[0])] = expected[j]
    return out, ev


def _forest_shell(trees, global_classes, n_features):
    """Create a lightweight sklearn RF shell around already-fitted trees for TreeSHAP."""
    C=len(global_classes)
    dummy_x=np.zeros((max(C,2), n_features), dtype=float)
    dummy_y=np.asarray(list(global_classes)[:max(C,2)]) if C >= 2 else np.array([global_classes[0],global_classes[0]])
    if len(dummy_y) != len(dummy_x):
        dummy_y=np.resize(np.asarray(global_classes), len(dummy_x))
    rf=RandomForestClassifier(n_estimators=1, random_state=SEED)
    try:
        rf.fit(dummy_x, dummy_y)
    except Exception:
        # For safety if dummy labels are not suitable, use a binary shell and replace classes below.
        rf.fit(np.zeros((2,n_features)), np.array([global_classes[0],global_classes[-1]]))
    rf.estimators_=list(trees)
    rf.n_estimators=len(trees)
    rf.classes_=np.asarray(global_classes)
    rf.n_classes_=len(global_classes)
    rf.n_features_in_=n_features
    return rf


def tree_union_shap(tree_specs, X, global_classes):
    """Compute TreeSHAP by exact fitted RF components and align their class labels.

    C2's pooled wrapper retains ``(tree, parent_forest.classes_)`` in ``tree_specs``.
    The underlying sklearn tree stores encoded numeric class positions, while the
    parent forest stores the real diagnosis/severity labels. We therefore explain
    each 10-tree parent component with TreeSHAP and then align its output columns
    using the recorded parent class labels.
    """
    specs=list(tree_specs); C=len(global_classes); p=X.shape[1]; n=len(X)
    vals=[np.zeros((n,p),dtype=float) for _ in range(C)]; exp=np.zeros(C,dtype=float)
    if not specs: return vals,exp
    groups=[]; start=0
    while start < len(specs):
        end=min(start+10,len(specs)); groups.append(specs[start:end]); start=end
    total=len(specs)
    for group in groups:
        parent_classes=np.asarray(group[0][1])
        trees=[pair[0] for pair in group]
        # The fitted sklearn trees encode their local class positions as 0..C-1.
        shell=_forest_shell(trees,np.arange(len(parent_classes)),p)
        ex=shap.TreeExplainer(shell)
        raw=ex.shap_values(X,check_additivity=True)
        if isinstance(raw,list): arr=[np.asarray(v,dtype=float) for v in raw]
        else:
            a=np.asarray(raw,dtype=float)
            arr=[a[:,:,c] for c in range(a.shape[2])] if a.ndim==3 else [a]
        ev=np.asarray(ex.expected_value,dtype=float).reshape(-1)
        weight=len(group)/total
        for j,c in enumerate(parent_classes):
            k=np.flatnonzero(np.asarray(global_classes)==c)
            if len(k) and j < len(arr): vals[int(k[0])] += arr[j]*weight
            if len(k) and j < len(ev): exp[int(k[0])] += ev[j]*weight
    return vals,exp

def prediction_and_shap(model, trees, X, labels):
    proba = model.predict_proba(X)
    pred = model.predict(X)
    vals, expected = tree_union_shap(trees, X, labels)
    errors=[]
    for c in range(len(labels)):
        errors.append(float(np.max(np.abs(expected[c] + vals[c].sum(axis=1) - proba[:, c]))))
    return pred, proba, vals, expected, max(errors)


def fit_personalization(X, y, n_trees, seed):
    params = dict(RF_PARAMS)
    params["n_estimators"] = int(n_trees)
    params["random_state"] = int(seed)
    return RandomForestClassifier(n_jobs=-1, **params).fit(X, y)


def preprocess_test_for_c2(df, assignments, features):
    work = df.drop(columns=[c for c in ["client_id", "split"] if c in df.columns], errors="ignore").merge(
        assignments, on="OASISID", how="left", validate="many_to_one")
    train = work[work.split.isin(["federation", "personalization"])].copy()
    test = work[work.split.eq("test")].copy()
    imp = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    Xtr = imp.fit_transform(train[features])
    Xte = imp.transform(test[features])
    trdf = pd.DataFrame(Xtr, columns=features, index=train.index)
    selected, _ = pearson_prune(trdf, features, threshold=PEARSON_THRESHOLD, verbose=False)
    # Use the model's stored selected feature order; all should be present.
    return test.reset_index(drop=True), pd.DataFrame(Xte, columns=features)[selected].reset_index(drop=True)


def reconstruct_task(task, root):
    task_dir = root / task
    import joblib
    pack = joblib.load(task_dir / "global_model.joblib")
    global_model = pack["global_model"]; features = list(pack["selected_features"]); labels = np.asarray(pack["labels"])
    assignments = pd.read_csv(task_dir / "sample_assignments.csv")
    if task == "diagnosis":
        df = pd.read_csv(RESULTS_DIR / "base_dataframe_unimputed.csv")
        label_col = "diagnosis_class"
    else:
        df = pd.read_csv(RESULTS_DIR / "severity_labelled_dataset.csv")
        label_col = "severity_label"
    test_meta, Xtest = preprocess_test_for_c2(df, assignments, features)
    # Reconstruct each private personalization forest using exactly the C2 assignment,
    # preprocessing philosophy and fixed seeds. No client test rows are used.
    work = df.merge(assignments, on="OASISID", how="left", validate="many_to_one")
    personal_models = {}; personalized_models = {}
    train_all = work[work.split.isin(["federation", "personalization"])].copy()
    imp = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    Xall_imp = imp.fit_transform(train_all[features])
    Xc_all = pd.DataFrame(Xall_imp, columns=features, index=train_all.index)
    # C2 fits one training-side imputer; reproduce it once, not once per client.
    for cid in range(10):
        c = work[(work.client_id == cid) & work.split.eq("personalization")]
        pos = train_all.index.isin(c.index)
        Xp0 = Xc_all.loc[pos, features]
        Xp, yp = smote_resample(Xp0.to_numpy(), c[label_col].to_numpy(), k_neighbors=5, random_state=SMOTE_RANDOM_STATE)
        lm = fit_personalization(Xp, yp, 10, SEED + cid)
        personal_models[cid] = lm
        personalized_models[cid] = PersonalizedRandomForest(global_model, lm)
    # Keep only actual client-test rows for explanation.
    test_meta = test_meta[test_meta["split"].eq("test")].reset_index(drop=True)
    Xtest = Xtest.loc[test_meta.index].reset_index(drop=True)
    test_meta["sample_id"] = [f"{sid}__row{j:05d}" for j, sid in enumerate(test_meta["OASISID"].astype(str))]
    return global_model, personal_models, personalized_models, test_meta, Xtest, features, labels, label_col


def select_samples(test_meta, Xtest, global_model, personalized_models, task):
    chosen=[]
    for cid in EXPLAIN_CLIENTS:
        idx = np.flatnonzero(test_meta.client_id.to_numpy() == cid)
        if len(idx)==0: continue
        rng=np.random.RandomState(SEED+cid)
        n=min(SAMPLES_PER_CLIENT,len(idx)); picks=rng.choice(idx,size=n,replace=False)
        chosen.extend(picks.tolist())
    return np.array(chosen,dtype=int)


def direction(shap_value):
    if shap_value > 0: return "toward predicted class"
    if shap_value < 0: return "away from predicted class"
    return "neutral"


def explain_sample(task, model_type, model, trees, Xrow, feature_names, labels, sample_id, client_id):
    pred, proba, vals, expected, err = prediction_and_shap(model, trees, Xrow, labels)
    cls_idx = int(np.flatnonzero(labels == pred[0])[0])
    sv = vals[cls_idx][0]
    fv = Xrow[0]
    rows=[]
    for f,v,s in zip(feature_names,fv,sv):
        rows.append({"sample_id":sample_id,"client_id":client_id,"task":task,"model_type":model_type,
                     "predicted_class":str(pred[0]),"predicted_probability":float(proba[0,cls_idx]),
                     "feature":f,"feature_value":float(v),"shap_value":float(s),"abs_shap_value":float(abs(s)),
                     "direction":direction(float(s)),"shap_additivity_error":float(err),"explained_class":str(pred[0])})
    return rows, pred[0], float(proba[0,cls_idx]), sv, err, proba[0]


def modality_aggregate(feature_rows, feature_names, modality_map):
    by={m:[] for m in MODALITY_ORDER}
    for row, f in zip(feature_rows, feature_names): by[modality_map.get(f,"UNMAPPED")].append(row)
    out=[]
    total=sum(abs(r["shap_value"]) for rs in by.values() for r in rs)
    for mod in MODALITY_ORDER:
        rs=by[mod]; abs_sum=sum(abs(r["shap_value"]) for r in rs); signed=sum(r["shap_value"] for r in rs)
        norm=abs_sum/total if total else np.nan
        out.append((mod,abs_sum,signed,norm))
    return out


def jaccard(a,b):
    a=set(a); b=set(b)
    return len(a&b)/len(a|b) if a|b else np.nan


def make_plots(out_dir, modality_df, patient_df, top_df, similarity_df):
    plots=Path(out_dir)/"plots"; plots.mkdir(parents=True,exist_ok=True)
    for task in ["diagnosis","severity"]:
        for model in ["global","personalized"]:
            d=modality_df[(modality_df.task==task)&(modality_df.model_type==model)&(modality_df.modality!="UNMAPPED")]
            if d.empty: continue
            s=d.groupby("modality").normalized_contribution.mean().reindex(MODALITY_ORDER[:3]).fillna(0)
            fig,ax=plt.subplots(figsize=(7,4)); s.plot(kind="bar",ax=ax); ax.set_ylabel("Mean normalized SHAP contribution"); ax.set_title(f"{task.title()} — {model.title()} modality contribution"); fig.tight_layout(); fig.savefig(plots/f"{task}_{model}_modality_contribution.png",dpi=160); plt.close(fig)
        g=modality_df[(modality_df.task==task)&(modality_df.model_type=="global")].groupby("modality").normalized_contribution.mean()
        p=modality_df[(modality_df.task==task)&(modality_df.model_type=="personalized")].groupby("modality").normalized_contribution.mean()
        comp=pd.DataFrame({"Global":g,"Personalized":p}).reindex(MODALITY_ORDER[:3])
        fig,ax=plt.subplots(figsize=(8,4)); comp.plot(kind="bar",ax=ax); ax.set_ylabel("Mean normalized SHAP contribution"); ax.set_title(f"{task.title()} — global vs personalized modalities"); fig.tight_layout(); fig.savefig(plots/f"{task}_global_vs_personalized_modality.png",dpi=160); plt.close(fig)
        c=modality_df[(modality_df.task==task)&(modality_df.model_type=="personalized")&modality_df.modality.ne("UNMAPPED")]
        if not c.empty:
            piv=c.pivot_table(index="client_id",columns="modality",values="normalized_contribution",aggfunc="mean").reindex(columns=MODALITY_ORDER[:3])
            if not piv.empty:
                fig,ax=plt.subplots(figsize=(9,5)); piv.plot(kind="bar",ax=ax); ax.set_ylabel("Mean normalized SHAP contribution"); ax.set_title(f"{task.title()} — personalized modality contribution for explained clients"); fig.tight_layout(); fig.savefig(plots/f"{task}_personalized_modality_by_client.png",dpi=160); plt.close(fig)
    # Feature-level summary bars: top mean |SHAP| across explained samples.
    for task in ["diagnosis","severity"]:
        d=patient_df[(patient_df.task==task)&(patient_df.model_type=="personalized")]
        if d.empty: continue
        top=d.groupby("feature").abs_shap_value.mean().sort_values(ascending=False).head(15).sort_values()
        fig,ax=plt.subplots(figsize=(8,6)); top.plot(kind="barh",ax=ax); ax.set_xlabel("Mean |SHAP|"); ax.set_title(f"{task.title()} — personalized feature SHAP summary"); fig.tight_layout(); fig.savefig(plots/f"{task}_personalized_feature_shap_summary.png",dpi=160); plt.close(fig)
    # Patient-level visualization: first selected sample, all modalities/models.
    if not modality_df.empty:
        r=modality_df.iloc[0]; mask=(modality_df.sample_id==r.sample_id)&(modality_df.client_id==r.client_id)&(modality_df.task==r.task)
        d=modality_df[mask & modality_df.modality.ne("UNMAPPED")].pivot(index="modality",columns="model_type",values="normalized_contribution").reindex(MODALITY_ORDER[:3]); fig,ax=plt.subplots(figsize=(8,4)); d.plot(kind="bar",ax=ax); ax.set_ylabel("Normalized SHAP contribution"); ax.set_title("Patient-level global vs personalized explanation"); fig.tight_layout(); fig.savefig(plots/f"patient_level_example_{r.task}_{r.client_id}_{r.sample_id}.png",dpi=160); plt.close(fig)


def run():
    root=RESULTS_DIR/"contribution2"; outroot=RESULTS_DIR/"contribution3"; outroot.mkdir(parents=True,exist_ok=True)
    config={"seed":SEED,"top_k_features_per_modality":TOP_K,"samples_per_client":SAMPLES_PER_CLIENT,"shap_package_version":shap.__version__,"additivity_tolerance":SHAP_ADD_TOL,"tasks":["diagnosis","severity"],"sample_selection":f"fixed random {SAMPLES_PER_CLIENT} client-test sample(s) per explained client, seed=42+client_id","note":"Post-hoc explanation only; no model retraining for explanation selection."}
    (outroot/"config").mkdir(exist_ok=True); (outroot/"config"/"contribution3_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    all_patient=[]; all_mod=[]; all_top=[]; all_sim=[]; all_client=[]; audit=[]; task_summ=[]; mapping_audit=[]
    for task in ["diagnosis","severity"]:
        tdir=outroot/task; tdir.mkdir(parents=True,exist_ok=True)
        global_model, personal_models, personalized_models, meta, Xtest, features, labels, label_col = reconstruct_task(task,root)
        mod_map, mod_status=modality_map_for(features)
        mapping_audit.extend({"task":task,"feature":f,"modality":mod_map[f],"mapping_status":mod_status[f]} for f in features)
        chosen=select_samples(meta,Xtest,global_model,personalized_models,task)
        global_trees=list(global_model.estimators_)
        global_specs=list(global_model.tree_specs)
        for idx in chosen:
            mrow=meta.iloc[idx]; cid=int(mrow.client_id); sid=mrow.sample_id; Xrow=Xtest.iloc[[idx]].to_numpy(dtype=float)
            for model_type, model, trees in [("global",global_model,global_specs),("personalized",personalized_models[cid],global_specs + [(t, np.asarray(personal_models[cid].classes_)) for t in personal_models[cid].estimators_]) ]:
                fr,pred,prob,sv,err,probs=explain_sample(task,model_type,model,trees,Xrow,features,labels,sid,cid)
                all_patient.extend(fr)
                agg=modality_aggregate(fr,features,mod_map)
                for mod,ab,sg,norm in agg:
                    all_mod.append({"sample_id":sid,"client_id":cid,"task":task,"model_type":model_type,"predicted_class":str(pred),"predicted_probability":prob,"modality":mod,"absolute_contribution":ab,"signed_contribution":sg,"normalized_contribution":norm,"shap_additivity_error":err})
                # top-K within modality
                for mod in MODALITY_ORDER:
                    rr=[r for r in fr if mod_map.get(r["feature"],"UNMAPPED")==mod]
                    rr=sorted(rr,key=lambda z:z["abs_shap_value"],reverse=True)[:TOP_K]
                    for rank,r in enumerate(rr,1):
                        all_top.append({"sample_id":sid,"client_id":cid,"task":task,"model_type":model_type,"predicted_class":str(pred),"modality":mod,"rank":rank,"feature":r["feature"],"feature_value":r["feature_value"],"shap_value":r["shap_value"],"abs_shap_value":r["abs_shap_value"]})
                audit.append({"task":task,"client_id":cid,"sample_id":sid,"model_type":model_type,"predicted_class":str(pred),"predicted_probability":prob,"max_additivity_error":err,"n_features":len(features),"n_global_trees":len(global_trees),"n_local_trees":0 if model_type=="global" else len(personal_models[cid].estimators_)})
            # global vs personalized comparisons for this sample
            gpred=next(a["predicted_class"] for a in audit[::-1] if a["task"]==task and a["sample_id"]==sid and a["client_id"]==cid and a["model_type"]=="global")
            ppred=next(a["predicted_class"] for a in audit[::-1] if a["task"]==task and a["sample_id"]==sid and a["client_id"]==cid and a["model_type"]=="personalized")
            gf=pd.DataFrame([r for r in all_patient if r["task"]==task and r["sample_id"]==sid and r["client_id"]==cid and r["model_type"]=="global"]).set_index("feature")
            pf=pd.DataFrame([r for r in all_patient if r["task"]==task and r["sample_id"]==sid and r["client_id"]==cid and r["model_type"]=="personalized"]).set_index("feature")
            for f in features:
                all_sim.append({"sample_id":sid,"client_id":cid,"task":task,"feature":f,"global_shap_value":float(gf.loc[f,"shap_value"]),"personalized_shap_value":float(pf.loc[f,"shap_value"]),"difference":float(pf.loc[f,"shap_value"]-gf.loc[f,"shap_value"]),"explained_class_match":bool(gpred==ppred)})
            if gpred==ppred:
                ga=gf.abs_shap_value.sort_values(ascending=False).index.tolist(); pa=pf.abs_shap_value.sort_values(ascending=False).index.tolist()
                
                gx=gf.loc[features,"abs_shap_value"].to_numpy(); px=pf.loc[features,"abs_shap_value"].to_numpy()
                if np.std(gx)==0 or np.std(px)==0:
                    rho=pv=np.nan
                else:
                    rho,pv=spearmanr(gx,px)
            else:
                ga=gf.abs_shap_value.sort_values(ascending=False).index.tolist(); pa=pf.abs_shap_value.sort_values(ascending=False).index.tolist(); rho=np.nan; pv=np.nan
            g5=ga[:5]; p5=pa[:5]; g10=ga[:10]; p10=pa[:10]
            gm={r["modality"]:r["normalized_contribution"] for r in all_mod if r["task"]==task and r["sample_id"]==sid and r["client_id"]==cid and r["model_type"]=="global"}
            pm={r["modality"]:r["normalized_contribution"] for r in all_mod if r["task"]==task and r["sample_id"]==sid and r["client_id"]==cid and r["model_type"]=="personalized"}
            validmods=[m for m in MODALITY_ORDER if m in gm and m in pm]
            gv=np.array([gm[m] for m in validmods]); pvmod=np.array([pm[m] for m in validmods]); mc=np.corrcoef(gv,pvmod)[0,1] if len(validmods)>1 and np.std(gv)>0 and np.std(pvmod)>0 else np.nan
            all_sim.append({"sample_id":sid,"client_id":cid,"task":task,"feature":"__SUMMARY__","global_shap_value":np.nan,"personalized_shap_value":np.nan,"difference":np.nan,"explained_class_match":bool(gpred==ppred),"global_predicted_class":str(gpred),"personalized_predicted_class":str(ppred),"spearman_correlation":float(rho) if np.isfinite(rho) else np.nan,"spearman_pvalue":float(pv) if np.isfinite(pv) else np.nan,"n_features_compared":len(features) if gpred==ppred else 0,"top5_overlap":jaccard(g5,p5),"top10_overlap":jaccard(g10,p10),"modality_correlation":mc})
        # outputs per task are filtered from global tables later
        task_summ.append({"task":task,"n_features":len(features),"n_explained_samples":len(chosen),"n_unmapped_features":sum(v=="UNMAPPED" for v in mod_map.values()),"features_by_modality":json.dumps({m:sum(v==m for v in mod_map.values()) for m in MODALITY_ORDER}),"max_additivity_error":max(a["max_additivity_error"] for a in audit if a["task"]==task)})
    patient=pd.DataFrame(all_patient); mod=pd.DataFrame(all_mod); top=pd.DataFrame(all_top); sim=pd.DataFrame(all_sim); aud=pd.DataFrame(audit)
    # summary similarity rows are the only rows with comparison statistics.
    sim_summary=sim[sim.feature.eq("__SUMMARY__")].copy()
    # client modality summaries
    # Client-level modality summaries use the existing C2 TreeSHAP aggregate (five
    # held-out client-test samples per client) and are therefore not re-computed by
    # the C3 patient-example pass. This preserves the existing C2 explanation audit.
    csum=[]
    mapping_df=pd.DataFrame(mapping_audit)
    for task in ["diagnosis","severity"]:
        imp_path=root/task/"shap_importance.csv"
        if imp_path.exists():
            imp=pd.read_csv(imp_path)
            for (cid,mt),g in imp.groupby(["client_id","model"]):
                for mod_name in MODALITY_ORDER[:3]:
                    fs=mapping_df[(mapping_df.task==task)&(mapping_df.modality==mod_name)]["feature"].tolist()
                    vals=g[g.feature.isin(fs)].mean_abs_shap
                    csum.append({"client_id":int(cid),"task":task,"model_type":mt,"modality":mod_name,
                                  "mean":float(vals.mean()) if len(vals) else np.nan,"std":float(vals.std(ddof=1)) if len(vals)>1 else 0.0,
                                  "median":float(vals.median()) if len(vals) else np.nan,"min":float(vals.min()) if len(vals) else np.nan,"max":float(vals.max()) if len(vals) else np.nan})
    cm=pd.DataFrame(csum)
    if not cm.empty:
        cm["normalized_contribution"] = cm["mean"] / cm.groupby(["client_id","task","model_type"])["mean"].transform("sum")
        cm["n_explained_samples"] = 5  # existing C2 SHAP audit uses five client-test samples per client
    # Write exact requested structure
    for task in ["diagnosis","severity"]:
        t=outroot/task
        patient[patient.task.eq(task)].to_csv(t/"patient_level_explanations.csv",index=False)
        mod[mod.task.eq(task)].to_csv(t/"modality_contributions.csv",index=False)
        top[top.task.eq(task)].to_csv(t/"top_features.csv",index=False)
        # feature-level comparison only, with prediction summary fields merged
        ss=sim_summary[sim_summary.task.eq(task)].copy()
        fs=sim[(sim.task.eq(task)) & (sim.feature.ne("__SUMMARY__"))].copy()
        fs_cols=["sample_id","client_id","task","feature","global_shap_value","personalized_shap_value","difference","explained_class_match"]
        fs.to_csv(t/"global_vs_personalized_explanations.csv",index=False,columns=fs_cols)
        ss.to_csv(t/"explanation_similarity.csv",index=False)
        cm[cm.task.eq(task)].to_csv(t/"client_modality_summary.csv",index=False)
        # machine-readable per-sample JSON files
        jdir=t/"patient_json"; jdir.mkdir(exist_ok=True)
        for (sid,cid),g in patient[patient.task.eq(task)].groupby(["sample_id","client_id"]):
            doc={"sample_id":sid,"client_id":int(cid),"task":task,"models":{}}
            for mt,mg in g.groupby("model_type"):
                pred=str(mg.iloc[0].predicted_class); prob=float(mg.iloc[0].predicted_probability); ae=float(mg.iloc[0].shap_additivity_error)
                md=mod[(mod.task==task)&(mod.sample_id==sid)&(mod.client_id==cid)&(mod.model_type==mt)]
                td=top[(top.task==task)&(top.sample_id==sid)&(top.client_id==cid)&(top.model_type==mt)]
                doc["models"][mt]={"predicted_class":pred,"predicted_probability":prob,"modalities":{},"shap_additivity_error":ae}
                for modality,mg2 in md.groupby("modality"):
                    tf=td[td.modality==modality].to_dict("records")
                    doc["models"][mt]["modalities"][modality]={"absolute_contribution":float(mg2.iloc[0].absolute_contribution),"signed_contribution":float(mg2.iloc[0].signed_contribution),"normalized_contribution":float(mg2.iloc[0].normalized_contribution) if np.isfinite(mg2.iloc[0].normalized_contribution) else None,"top_features":tf}
            (jdir/f"{sid}_client_{int(cid):02d}.json").write_text(json.dumps(doc,indent=2,default=str),encoding="utf-8")
    (outroot/"summary").mkdir(exist_ok=True)
    pd.DataFrame(task_summ).to_csv(outroot/"summary"/"contribution3_summary.csv",index=False)
    mod.groupby(["task","model_type","modality"])["normalized_contribution"].agg(["mean","std","median","min","max"]).reset_index().to_csv(outroot/"summary"/"modality_summary.csv",index=False)
    sim_summary.to_csv(outroot/"summary"/"explanation_similarity_summary.csv",index=False)
    pd.DataFrame(mapping_audit).to_csv(outroot/"summary"/"feature_modality_mapping_audit.csv",index=False)
    aud.to_csv(outroot/"summary"/"shap_audit.csv",index=False)
    make_plots(outroot,mod,patient,top,sim_summary)
    # Per-client modality plot based on the existing C2 TreeSHAP aggregate (all 10 clients).
    for task in ["diagnosis","severity"]:
        d=cm[cm.task.eq(task) & cm.modality.ne("UNMAPPED")].copy()
        if not d.empty:
            piv=d[d.model_type.eq("personalized")].pivot_table(index="client_id",columns="modality",values="normalized_contribution",aggfunc="mean").reindex(columns=MODALITY_ORDER[:3])
            if not piv.empty:
                fig,ax=plt.subplots(figsize=(10,5)); piv.plot(kind="bar",ax=ax); ax.set_ylabel("Normalized SHAP contribution"); ax.set_title(f"{task.title()} — personalized modality contribution by client"); fig.tight_layout(); (outroot/task/"plots").mkdir(parents=True,exist_ok=True); fig.savefig(outroot/task/"plots"/f"{task}_all_client_personalized_modality.png",dpi=160); plt.close(fig)
    print(pd.DataFrame(task_summ).to_string(index=False))
    print("MAX SHAP ERROR", aud.max_additivity_error.max())
    return outroot

if __name__ == "__main__":
    run()
