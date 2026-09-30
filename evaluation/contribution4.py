"""Contribution 4: computational, communication, and latency evaluation.

This module is an instrumentation/evaluation layer. It does not replace or
modify the Base Paper, C1, C2, or C3 algorithms. It reconstructs the same C2
participant/client experiment deterministically, measures the existing
round-wise RF tree-pooling operations, and compares them with a centralized
RF baseline trained on the same federation-training rows.

All durations use time.perf_counter(). Serialized payloads are measured with
joblib's actual byte stream. Network latency is deliberately NOT measured:
the current FL implementation executes in one Python process.
"""
from __future__ import annotations
import io, json, os, platform, sys, time, statistics

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn, shap
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.model_config import RESULTS_DIR, RF_PARAMS, KNN_NEIGHBORS, PEARSON_THRESHOLD, SMOTE_RANDOM_STATE
from preprocessing.preprocess_common import pearson_prune, smote_resample
from models.random_forest import train_rf
from federated.personalized_rf import train_non_iid_federated, personalize_global_model, PooledRandomForest
from evaluation.contribution2 import (
    SEED, N_CLIENTS, DIRICHLET_ALPHA, MIN_CLIENT_SAMPLES,
    PERSONALIZATION_TREES, participant_train_test, dirichlet_subject_assignment,
    split_client_subjects, preprocess_for_c2,
)
from explainability.contribution3 import reconstruct_task, select_samples, explain_sample

OUT = RESULTS_DIR / "contribution4"
CONFIG_DIR = OUT / "config"
PLOTS = OUT / "plots"
for p in [OUT, CONFIG_DIR, PLOTS, OUT/"centralized", OUT/"federated", OUT/"personalized", OUT/"shap", OUT/"summary"]: p.mkdir(parents=True, exist_ok=True)

N_ROUNDS = 50
TREES_PER_CLIENT = 10
GLOBAL_TREES = 100
TOP_K = 5
TIMING_REPETITIONS = 7
INFERENCE_REPETITIONS = 20


def monotonic(): return time.perf_counter()


def rss_bytes() -> int | None:
    """Current RSS where procfs is available; otherwise None."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


# def process_rss_bytes() -> int | None:
#     """Process peak RSS. Linux ru_maxrss is KiB; macOS is bytes."""
#     try:
#         v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#         if sys.platform == "darwin": return int(v)
#         return int(v) * 1024
#     except Exception:
#         return None

def process_rss_bytes() -> int | None:
    """Current process RSS snapshot in bytes; not a stage-isolated peak."""
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None

def serialize_bytes(obj) -> tuple[bytes, float]:
    bio = io.BytesIO()
    t0 = monotonic()
    joblib.dump(obj, bio, compress=0, protocol=5)
    dt = monotonic() - t0
    return bio.getvalue(), dt


def serialize_size(obj) -> tuple[int, float]:
    b, dt = serialize_bytes(obj)
    return len(b), dt


def tree_payload(forest) -> tuple[int, float]:
    return serialize_size(forest)


def safe_float(x):
    return None if x is None or not np.isfinite(x) else float(x)


def timing_stats(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if len(a)>1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def build_c2_task(task: str):
    """Reproduce C2's exact preprocessing/client partition without modifying C2."""
    if task == "diagnosis":
        df = pd.read_csv(RESULTS_DIR / "base_dataframe_unimputed.csv")
        label_col = "diagnosis_class"
        feature_cols = [c for c in df.columns if c not in {"OASISID", label_col, "MRI_matched"}]
    elif task == "severity":
        df = pd.read_csv(RESULTS_DIR / "severity_labelled_dataset.csv")
        label_col = "severity_label"
        feature_cols = [c for c in df.columns if c not in {"OASISID", label_col, "CDRTOT_source"}]
    else: raise ValueError(task)

    tr_idx, outer_test_idx = participant_train_test(df)
    work = df.iloc[tr_idx].reset_index(drop=True).copy()
    outer_test = df.iloc[outer_test_idx].reset_index(drop=True).copy()
    assigned, _ = dirichlet_subject_assignment(work, label_col, seed=SEED)
    counts = assigned.groupby("client_id").size()
    if len(counts) != N_CLIENTS or int(counts.min()) < MIN_CLIENT_SAMPLES:
        raise RuntimeError(f"C2 client allocation mismatch: {counts.to_dict()}")
    subject_client = assigned[["OASISID", "client_id"]].drop_duplicates("OASISID")
    split_map = split_client_subjects(subject_client, seed=SEED)
    assigned = split_map[["OASISID", "client_id", "split"]].copy()
    work2 = work.merge(assigned, on="OASISID", how="left", validate="many_to_one")
    train_proc, test_proc, selected, dropped = preprocess_for_c2(work2, label_col, feature_cols, assigned)
    client_data = {}
    fed_shards = []
    for cid in range(N_CLIENTS):
        ctrain = train_proc[(train_proc.client_id==cid)&(train_proc.split=="federation")]
        cpers = train_proc[(train_proc.client_id==cid)&(train_proc.split=="personalization")]
        ctest = test_proc[test_proc.client_id==cid]
        Xf, yf = smote_resample(ctrain[selected].to_numpy(), ctrain[label_col].to_numpy(), k_neighbors=5, random_state=SMOTE_RANDOM_STATE)
        Xp, yp = smote_resample(cpers[selected].to_numpy(), cpers[label_col].to_numpy(), k_neighbors=5, random_state=SMOTE_RANDOM_STATE)
        fed_shards.append((Xf, yf))
        client_data[cid] = {"Xf":Xf,"yf":yf,"Xp":Xp,"yp":yp,"ctest":ctest,
                            "n_fed_subjects":int(ctrain.OASISID.nunique()),
                            "n_personal_subjects":int(cpers.OASISID.nunique()),
                            "n_test_subjects":int(ctest.OASISID.nunique())}
    labels = np.sort(train_proc[label_col].unique())
    return dict(task=task, df=df, label_col=label_col, features=selected, dropped=dropped,
                work=work, outer_test=outer_test, train_proc=train_proc, test_proc=test_proc,
                client_data=client_data, fed_shards=fed_shards, labels=labels)


def benchmark_centralized(ctx):
    # C4 baseline: same federation-training data as global FL, pooled centrally.
    X = np.vstack([ctx["client_data"][cid]["Xf"] for cid in range(N_CLIENTS)])
    y = np.concatenate([ctx["client_data"][cid]["yf"] for cid in range(N_CLIENTS)])
    model_sizes=[]; train_times=[]; cpu_times=[]
    for _ in range(TIMING_REPETITIONS):
        t0=monotonic(); c0=time.process_time(); model=train_rf(X,y); train_times.append(monotonic()-t0); cpu_times.append(time.process_time()-c0)
    # Use the final measured model; repeat inference on the combined held-out client tests.
    Xtest=np.vstack([ctx["client_data"][cid]["ctest"][ctx["features"]].to_numpy() for cid in range(N_CLIENTS)])
    ytest=np.concatenate([ctx["client_data"][cid]["ctest"][ctx["label_col"]].to_numpy() for cid in range(N_CLIENTS)])
    pred=model.predict(Xtest)
    size,_=serialize_size(model)
    single=Xtest[:min(64,len(Xtest))]
    # Same protocol for all models: warmed, repeated calls; batch uses full client test set.
    _=model.predict(single)
    one=[]; batch=[]
    for _ in range(INFERENCE_REPETITIONS):
        t=monotonic(); model.predict(single[0:1]); one.append(monotonic()-t)
        t=monotonic(); model.predict(single); batch.append(monotonic()-t)
    out={"system":"centralized","task":ctx["task"],"training_time_mean":timing_stats(train_times)["mean"],
         "training_time_median":timing_stats(train_times)["median"],"training_time_std":timing_stats(train_times)["std"],
         "training_time_min":timing_stats(train_times)["min"],"training_time_max":timing_stats(train_times)["max"],
         "cpu_process_time_mean":float(np.mean(cpu_times)),"model_size_bytes":size,
         "inference_single_mean":float(np.mean(one)),"inference_single_median":float(np.median(one)),
         "inference_batch_mean":float(np.mean(batch)),"inference_batch_median":float(np.median(batch)),
         "n_train":len(y),"n_test":len(ytest),"process_rss_snapshot_bytes":process_rss_bytes(),"communication_bytes":np.nan}
    pd.DataFrame([out]).to_csv(OUT/"centralized"/f"{ctx['task']}_metrics.csv",index=False)
    return model,out


def benchmark_federated(ctx):
    task=ctx["task"]; shards=ctx["fed_shards"]
    rows=[]; client_rows=[]; cumulative_up=0; cumulative_down=0
    global_model=None
    # Exact C2 tree-pooling implementation, but with timing wrapped around each operation.
    from federated.federated_rf import _tree_budget, _fit_client_forest
    budgets=_tree_budget(N_CLIENTS, GLOBAL_TREES)
    fl_t0=monotonic(); fl_cpu0=time.process_time(); 
    for rnd in range(1,N_ROUNDS+1):
        r0=monotonic(); client_models=[]; upload_total=0; client_train_total=0; ser_total=0
        round_client=[]
        for cid,((Xc,yc),nt) in enumerate(zip(shards,budgets)):
            t=monotonic(); c0=time.process_time()
            forest=_fit_client_forest(Xc,yc,nt,SEED + rnd*10000 + cid)
            train_dt=monotonic()-t; cpu_dt=time.process_time()-c0
            payload,ser_dt=serialize_bytes(forest); upload=len(payload)
            client_models.append(forest); upload_total += upload; client_train_total += train_dt; ser_total += ser_dt
            round_client.append((cid,train_dt,ser_dt,upload,cpu_dt,forest))
        # C2 uses PooledRandomForest for non-IID class alignment; reproduce that exact construction.
        t=monotonic()
        global_model=PooledRandomForest(client_models, np.unique(np.concatenate([np.asarray(y) for _,y in shards])))
        agg_dt=monotonic()-t
        t=monotonic(); gpayload,gser=serialize_bytes(global_model); download=len(gpayload); gser_dt=gser
        round_ser=ser_total+gser_dt
        cumulative_up += upload_total; cumulative_down += download
        total_dt=monotonic()-r0
        rows.append({"round":rnd,"client_training_time":client_train_total,"serialization_time":round_ser,
                     "upload_bytes":upload_total,"server_aggregation_time":agg_dt,"global_serialization_time":gser_dt,
                     "simulated_broadcast_bytes":download,"round_total_time":total_dt,
                     "simulated_total_communication":upload_total+download,
                     "cumulative_upload_bytes":cumulative_up,"cumulative_simulated_broadcast_bytes":cumulative_down,
                     "cumulative_simulated_total_bytes":cumulative_up+cumulative_down,"process_rss_snapshot_bytes":process_rss_bytes()})
        for cid,td,sd,up,cp,forest in round_client:
            client_rows.append({"round":rnd,"client_id":cid,"client_training_time":td,"serialization_time":sd,
                                "upload_bytes":up,"simulated_broadcast_bytes":download,"client_cpu_time":cp,
                                "server_aggregation_time":agg_dt,"round_total_time":total_dt})
    fl_time=monotonic()-fl_t0; fl_cpu=time.process_time()-fl_cpu0
    # Final prediction latency on all controlled client-test samples.
    Xtest_by_client={cid:ctx["client_data"][cid]["ctest"][ctx["features"]].to_numpy() for cid in range(N_CLIENTS)}
    alltest=np.vstack([Xtest_by_client[c] for c in range(N_CLIENTS)])
    _=global_model.predict(alltest[:64])
    one=[]; batch=[]
    for _ in range(INFERENCE_REPETITIONS):
        t=monotonic(); global_model.predict(alltest[:1]); one.append(monotonic()-t)
        t=monotonic(); global_model.predict(alltest[:64]); batch.append(monotonic()-t)
    model_size,_=serialize_size(global_model)
    round_df=pd.DataFrame(rows); client_df=pd.DataFrame(client_rows)
    base=OUT/"federated"; base.mkdir(parents=True,exist_ok=True)
    round_df.to_csv(base/f"{task}_round_metrics.csv",index=False); client_df.to_csv(base/f"{task}_client_metrics.csv",index=False)
    comm=round_df[["round","upload_bytes","simulated_broadcast_bytes","simulated_total_communication","cumulative_upload_bytes","cumulative_simulated_broadcast_bytes","cumulative_simulated_total_bytes"]].copy()
    comm["average_bytes_per_client_upload"] = comm.upload_bytes/N_CLIENTS
    comm["average_bytes_per_client_round_total"] = comm.simulated_total_communication/N_CLIENTS
    comm.to_csv(base/f"{task}_communication_metrics.csv",index=False)
    summary={"system":"global_federated","task":task,"training_time_total":fl_time,"cpu_process_time_total":fl_cpu,
             "avg_round_time":round_df.round_total_time.mean(),"median_round_time":round_df.round_total_time.median(),
             "std_round_time":round_df.round_total_time.std(ddof=1),"min_round_time":round_df.round_total_time.min(),"max_round_time":round_df.round_total_time.max(),
             "total_client_training_time":round_df.client_training_time.sum(),"total_aggregation_time":round_df.server_aggregation_time.sum(),
             "total_serialization_time":round_df.serialization_time.sum(),"total_upload_bytes":round_df.upload_bytes.sum(),
             "total_simulated_broadcast_bytes":round_df.simulated_broadcast_bytes.sum(),"total_simulated_communication_bytes":round_df.simulated_total_communication.sum(),
             "avg_simulated_communication_per_round":round_df.simulated_total_communication.mean(),"avg_simulated_communication_per_client_round":round_df.simulated_total_communication.mean()/N_CLIENTS,
             "model_size_bytes":model_size,"inference_single_mean":np.mean(one),"inference_single_median":np.median(one),
             "inference_batch_mean":np.mean(batch),"inference_batch_median":np.median(batch),"process_rss_snapshot_bytes":process_rss_bytes(),"network_latency":"NOT MEASURED"}
    pd.DataFrame([summary]).to_csv(base/f"{task}_metrics.csv",index=False)
    return global_model,summary,round_df,client_df


def benchmark_personalized(ctx, global_model):
    task=ctx["task"]; rows=[]; models={}; locals_={}
    t_global_start=monotonic()
    # Do not retrain global here; this is measured from the already executed global C4 run.
    global_size,_=serialize_size(global_model)
    for cid in range(N_CLIENTS):
        d=ctx["client_data"][cid]; t=monotonic(); c0=time.process_time()
        personalized, local_model=personalize_global_model(global_model,d["Xp"],d["yp"],n_trees=PERSONALIZATION_TREES,seed=SEED+cid)
        ptrain=monotonic()-t; pcpu=time.process_time()-c0
        local_size,local_ser=serialize_size(local_model); combined_size,combined_ser=serialize_size(personalized)
        Xtest=d["ctest"][ctx["features"]].to_numpy()
        _=personalized.predict(Xtest[:min(16,len(Xtest))])
        one=[]; batch=[]
        for _ in range(INFERENCE_REPETITIONS):
            t=monotonic(); personalized.predict(Xtest[:1]); one.append(monotonic()-t)
            t=monotonic(); personalized.predict(Xtest[:min(64,len(Xtest))]); batch.append(monotonic()-t)
        rows.append({"client_id":cid,"personalization_time":ptrain,"personalization_cpu_time":pcpu,
                     "local_tree_serialization_time":local_ser,"personalization_tree_size_bytes":local_size,
                     "global_model_size_bytes":global_size,"personalized_model_size_bytes":combined_size,
                     "personalization_model_size_overhead_bytes":combined_size-global_size,
                     "personalization_model_size_ratio":combined_size/global_size,
                     "inference_single_mean":np.mean(one),"inference_single_median":np.median(one),
                     "inference_batch_mean":np.mean(batch),"inference_batch_median":np.median(batch),
                     "process_rss_snapshot_after_personalization_bytes":process_rss_bytes(),"global_tree_count":len(global_model.estimators_),
                     "local_tree_count":len(local_model.estimators_),"personalized_tree_count":len(personalized.estimators_)})
        models[cid]=personalized; locals_[cid]=local_model
    df=pd.DataFrame(rows); base=OUT/"personalized"; base.mkdir(parents=True,exist_ok=True)
    df.to_csv(base/"client_metrics.csv",index=False)
    df.to_csv(base/f"{task}_client_metrics.csv",index=False)
    # overall model-size view, one representative row per model family plus per-client personalization.
    size_rows=[{"task":task,"system":"global_federated","trees":GLOBAL_TREES,"serialized_size_bytes":global_size,"additional_local_trees":0,"overhead_bytes":0}]
    for r in rows:
        size_rows.append({"task":task,"system":"personalized_client","client_id":r["client_id"],"trees":r["personalized_tree_count"],"serialized_size_bytes":r["personalized_model_size_bytes"],"additional_local_trees":r["local_tree_count"],"overhead_bytes":r["personalization_model_size_overhead_bytes"]})
    pd.DataFrame(size_rows).to_csv(base/"model_size_metrics.csv",index=False)
    pd.DataFrame(size_rows).to_csv(base/f"{task}_model_size_metrics.csv",index=False)
    # Aggregate task metrics.
    pd.DataFrame([{"system":"personalized_federated","task":task,
                    "global_fl_time_excluded_from_personalization":np.nan,
                    "personalization_time_mean":df.personalization_time.mean(),"personalization_time_total":df.personalization_time.sum(),
                    "personalization_time_std":df.personalization_time.std(ddof=1),"personalization_time_min":df.personalization_time.min(),"personalization_time_max":df.personalization_time.max(),
                    "personalized_model_size_mean":df.personalized_model_size_bytes.mean(),"personalized_model_size_std":df.personalized_model_size_bytes.std(ddof=1),
                    "personalization_size_overhead_mean":df.personalization_model_size_overhead_bytes.mean(),"personalization_size_ratio_mean":df.personalization_model_size_ratio.mean(),
                    "inference_single_mean":df.inference_single_mean.mean(),"inference_batch_mean":df.inference_batch_mean.mean()}]).to_csv(base/f"{task}_metrics.csv",index=False)
    return models,locals_,df


def benchmark_shap(task, ctx, global_model, personal_models):
    # Reuse C3's exact sample-selection methodology and actual TreeSHAP implementation.
    root=RESULTS_DIR/"contribution2"
    c3_global,c3_personal,c3_personalized,meta,Xtest,features,labels,label_col = reconstruct_task(task,root)
    chosen=select_samples(meta,Xtest,c3_global,c3_personalized,task)
    rows=[]
    # Use C3 reconstructed models, which are the serialized C2 models and exact personalization models.
    for idx in chosen:
        mrow=meta.iloc[idx]; cid=int(mrow.client_id); sid=mrow.sample_id; Xrow=Xtest.iloc[[idx]].to_numpy(dtype=float)
        for model_type,model,trees in [
            ("global",c3_global,list(c3_global.tree_specs)),
            ("personalized",c3_personalized[cid],list(c3_global.tree_specs)+[(t,np.asarray(c3_personal[cid].classes_)) for t in c3_personal[cid].estimators_])]:
            # Prediction-only repeated once per benchmark sample; then exact C3 explain_sample.
            _=model.predict(Xrow)
            pred_times=[]; shap_times=[]; total_times=[]
            for _ in range(TIMING_REPETITIONS):
                t=monotonic(); model.predict(Xrow); pred_dt=monotonic()-t
                t=monotonic(); explain_sample(task,model_type,model,trees,Xrow,features,labels,sid,cid); shap_dt=monotonic()-t
                pred_times.append(pred_dt); shap_times.append(shap_dt); total_times.append(pred_dt+shap_dt)
            rows.append({"task":task,"model_type":model_type,"client_id":cid,"sample_id":sid,
                         "prediction_time_mean":np.mean(pred_times),"prediction_time_median":np.median(pred_times),"prediction_time_std":np.std(pred_times,ddof=1),
                         "prediction_plus_shap_time_mean":np.mean(total_times),"shap_overhead_mean":np.mean(shap_times),
                         "shap_overhead_median":np.median(shap_times),"shap_overhead_std":np.std(shap_times,ddof=1),
                         "shap_over_prediction_ratio":np.mean(shap_times)/np.mean(pred_times),"repetitions":TIMING_REPETITIONS})
    df=pd.DataFrame(rows); base=OUT/"shap"; base.mkdir(parents=True,exist_ok=True)
    df.to_csv(base/f"{task}_latency.csv",index=False)
    return df


def plot_outputs(task_results):
    import matplotlib.pyplot as plt
    # Training comparison
    rows=[]
    for task, r in task_results.items():
        rows += [{"task":task,"system":"Centralized","value":r["central"][1]["training_time_mean"]},
                 {"task":task,"system":"Global FL","value":r["fed"][1]["training_time_total"]},
                 {"task":task,"system":"Personalized FL","value":r["fed"][1]["training_time_total"]+r["pers"][2].personalization_time.sum()}]
    d=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(9,5)); d.pivot(index="system",columns="task",values="value").plot(kind="bar",ax=ax); ax.set_ylabel("Training time (s)"); ax.set_title("Centralized vs federated vs personalized training time"); fig.tight_layout(); fig.savefig(PLOTS/"training_time_comparison.png",dpi=160); plt.close(fig)
    # Model sizes
    rows=[]
    for task,r in task_results.items():
        cent_size=r["central"][1]["model_size_bytes"]; glob_size=r["fed"][1]["model_size_bytes"]; pers_mean=r["pers"][2].personalized_model_size_bytes.mean()
        rows += [{"task":task,"system":"Centralized RF","bytes":cent_size},{"task":task,"system":"Global FL RF","bytes":glob_size},{"task":task,"system":"Personalized RF (mean)","bytes":pers_mean}]
    d=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(9,5)); d.pivot(index="system",columns="task",values="bytes").plot(kind="bar",ax=ax); ax.set_ylabel("Serialized model size (bytes)"); ax.set_title("Serialized model size comparison"); fig.tight_layout(); fig.savefig(PLOTS/"model_size_comparison.png",dpi=160); plt.close(fig)
    # Inference
    rows=[]
    for task,r in task_results.items():
        rows += [{"task":task,"system":"Centralized","value":r["central"][1]["inference_single_mean"]},{"task":task,"system":"Global FL","value":r["fed"][1]["inference_single_mean"]},{"task":task,"system":"Personalized","value":r["pers"][2].inference_single_mean.mean()}]
    d=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(9,5)); d.pivot(index="system",columns="task",values="value").plot(kind="bar",ax=ax); ax.set_ylabel("Single-sample inference latency (s)"); ax.set_title("Inference latency comparison"); fig.tight_layout(); fig.savefig(PLOTS/"inference_latency_comparison.png",dpi=160); plt.close(fig)
    # SHAP
    shap_all=[]
    for task,r in task_results.items():
        sdf=r["shap"]
        for mt,g in sdf.groupby("model_type"): shap_all.append({"task":task,"system":mt,"value":g.shap_overhead_mean.mean()})
    d=pd.DataFrame(shap_all); fig,ax=plt.subplots(figsize=(9,5)); d.pivot(index="system",columns="task",values="value").plot(kind="bar",ax=ax); ax.set_ylabel("SHAP explanation overhead (s)"); ax.set_title("TreeSHAP explanation latency"); fig.tight_layout(); fig.savefig(PLOTS/"shap_latency_comparison.png",dpi=160); plt.close(fig)
    # Round plots and cumulative communication per task.
    for task,r in task_results.items():
        rd=r["fed"][2]
        fig,ax=plt.subplots(figsize=(10,5)); ax.plot(rd["round"],rd["round_total_time"],label="Total round"); ax.plot(rd["round"],rd["client_training_time"],label="Client training"); ax.plot(rd["round"],rd["server_aggregation_time"],label="Aggregation"); ax.plot(rd["round"],rd["serialization_time"],label="Serialization"); ax.set_xlabel("FL round"); ax.set_ylabel("Time (s)"); ax.set_title(f"{task.title()} — FL round duration components"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS/f"round_latency_{task}.png",dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5)); ax.plot(rd["round"],rd["cumulative_upload_bytes"],label="Cumulative upload"); ax.plot(rd["round"],rd["cumulative_simulated_broadcast_bytes"],label="Cumulative simulated broadcast payload"); ax.plot(rd["round"],rd["cumulative_simulated_total_bytes"],label="Cumulative total"); ax.set_xlabel("FL round"); ax.set_ylabel("Bytes"); ax.set_title(f"{task.title()} — cumulative serialized communication"); ax.legend(); fig.tight_layout(); fig.savefig(PLOTS/f"cumulative_communication_{task}.png",dpi=160); plt.close(fig)
        cd=r["fed"][3]
        fig,ax=plt.subplots(figsize=(9,5)); cd.groupby("client_id").client_training_time.mean().plot(kind="bar",ax=ax); ax.set_xlabel("Client"); ax.set_ylabel("Mean training time (s)"); ax.set_title(f"{task.title()} — client training time"); fig.tight_layout(); fig.savefig(PLOTS/f"client_training_time_{task}.png",dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5)); cd.groupby("client_id").upload_bytes.mean().plot(kind="bar",ax=ax); ax.set_xlabel("Client"); ax.set_ylabel("Mean upload bytes/round"); ax.set_title(f"{task.title()} — client communication"); fig.tight_layout(); fig.savefig(PLOTS/f"client_communication_{task}.png",dpi=160); plt.close(fig)
    # Required generic filenames: diagnosis/severity are preserved above; use diagnosis as representative only if both exist.
    # The task-specific plots are the authoritative ones.


def make_summaries(task_results):
    rows=[]; comm=[]; mem=[]; inf=[]; lat=[]; pers=[]; shaprows=[]
    for task,r in task_results.items():
        c=r["central"][1]; f=r["fed"][1]; p=r["pers"][2]; s=r["shap"]
        rows += [
          {"system":"centralized","task":task,"training_time":c["training_time_mean"],"avg_round_time":np.nan,"client_training_time":np.nan,"aggregation_time":np.nan,"communication_upload_bytes":np.nan,"simulated_broadcast_bytes":np.nan,"simulated_communication_bytes":np.nan,"model_size_bytes":c["model_size_bytes"],"memory_rss_snapshot_bytes":c["process_rss_snapshot_bytes"],"inference_time":c["inference_single_mean"],"shap_time":np.nan},
          {"system":"global_federated","task":task,"training_time":f["training_time_total"],"avg_round_time":f["avg_round_time"],"client_training_time":f["total_client_training_time"],"aggregation_time":f["total_aggregation_time"],"communication_upload_bytes":f["total_upload_bytes"],"simulated_broadcast_bytes":f["total_simulated_broadcast_bytes"],"simulated_communication_bytes":f["total_simulated_communication_bytes"],"model_size_bytes":f["model_size_bytes"],"memory_rss_snapshot_bytes":f["process_rss_snapshot_bytes"],"inference_time":f["inference_single_mean"],"shap_time":s[s.model_type.eq('global')].shap_overhead_mean.mean()},
          {"system":"personalized_federated","task":task,"training_time":f["training_time_total"]+p.personalization_time.sum(),"avg_round_time":f["avg_round_time"],"client_training_time":f["total_client_training_time"],"aggregation_time":f["total_aggregation_time"],"communication_upload_bytes":f["total_upload_bytes"],"simulated_broadcast_bytes":f["total_simulated_broadcast_bytes"],"simulated_communication_bytes":f["total_simulated_communication_bytes"],"model_size_bytes":p.personalized_model_size_bytes.mean(),"memory_rss_snapshot_bytes":np.nan,"inference_time":p.inference_single_mean.mean(),"shap_time":s[s.model_type.eq('personalized')].shap_overhead_mean.mean()},
        ]
        comm.append({"task":task,"total_upload_bytes":f["total_upload_bytes"],"total_simulated_broadcast_bytes":f["total_simulated_broadcast_bytes"],"total_simulated_communication_bytes":f["total_simulated_communication_bytes"],"average_bytes_per_round":f["avg_simulated_communication_per_round"],"average_bytes_per_client_round":f["avg_simulated_communication_per_client_round"]})
        mem.append({"task":task,"centralized_process_rss_snapshot_bytes":c["process_rss_snapshot_bytes"],"global_federated_process_rss_snapshot_bytes":f["process_rss_snapshot_bytes"],"personalized_process_rss_snapshot_bytes":np.nan})
        inf += [{"task":task,"system":"centralized","single_sample_seconds":c["inference_single_mean"],"batch_seconds":c["inference_batch_mean"]},{"task":task,"system":"global_federated","single_sample_seconds":f["inference_single_mean"],"batch_seconds":f["inference_batch_mean"]},{"task":task,"system":"personalized_federated","single_sample_seconds":p.inference_single_mean.mean(),"batch_seconds":p.inference_batch_mean.mean()}]
        lat.append({"task":task,"centralized_training_time":c["training_time_mean"],"federated_training_time":f["training_time_total"],"avg_round_time":f["avg_round_time"],"median_round_time":f["median_round_time"],"std_round_time":f["std_round_time"],"min_round_time":f["min_round_time"],"max_round_time":f["max_round_time"],"total_aggregation_time":f["total_aggregation_time"],"total_serialization_time":f["total_serialization_time"]})
        pers.append({"task":task,"personalization_time_mean":p.personalization_time.mean(),"personalization_time_total":p.personalization_time.sum(),"personalization_time_over_global_fl":p.personalization_time.sum()/f["training_time_total"],"personalized_model_size_mean":p.personalized_model_size_bytes.mean(),"global_model_size":f["model_size_bytes"],"personalized_size_over_global":p.personalized_model_size_bytes.mean()/f["model_size_bytes"],"federated_over_centralized_training_time":f["training_time_total"]/c["training_time_mean"]})
        for mt,g in s.groupby("model_type"):
            shaprows.append({"task":task,"model_type":mt,"n_explained_samples":len(g),"mean_prediction_time":g.prediction_time_mean.mean(),"mean_prediction_plus_shap":g.prediction_plus_shap_time_mean.mean(),"mean_shap_overhead":g.shap_overhead_mean.mean(),"median_shap_overhead":g.shap_overhead_median.mean(),"std_shap_overhead":g.shap_overhead_std.mean(),"shap_over_prediction_ratio":g.shap_over_prediction_ratio.mean()})
    summary=OUT/"summary"; summary.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(summary/"training_time_comparison.csv",index=False)
    pd.DataFrame(comm).to_csv(summary/"communication_comparison.csv",index=False)
    pd.DataFrame(mem).to_csv(summary/"memory_comparison.csv",index=False)
    pd.DataFrame(inf).to_csv(summary/"inference_comparison.csv",index=False)
    pd.DataFrame(lat).to_csv(summary/"latency_comparison.csv",index=False)
    pd.DataFrame(pers).to_csv(summary/"personalization_overhead.csv",index=False)
    pd.DataFrame(shaprows).to_csv(summary/"shap_overhead.csv",index=False)
    pd.DataFrame(rows).to_csv(summary/"master_summary.csv",index=False)


def environment():
    try: import psutil; cpu=psutil.cpu_count(logical=True); ram=psutil.virtual_memory().total
    except Exception: cpu=ram=None
    env={"python_version":platform.python_version(),"scikit_learn_version":sklearn.__version__,"shap_version":shap.__version__,
         "cpu_logical_count":cpu,"ram_bytes":ram,"operating_system":platform.system()+" "+platform.release(),
         "rf_configuration":RF_PARAMS,"n_clients":N_CLIENTS,"n_rounds":N_ROUNDS,"trees_per_client":TREES_PER_CLIENT,
         "global_trees":GLOBAL_TREES,"personalization_trees":PERSONALIZATION_TREES,"seed":SEED,
         "timing_method":"time.perf_counter","timing_repetitions":TIMING_REPETITIONS,"inference_repetitions":INFERENCE_REPETITIONS,
         "memory_method":"process RSS snapshots; these are not stage-isolated peak-memory measurements",
         "network_latency":"NOT MEASURED","network_bandwidth":"NOT MEASURED",
         "communication_definition":"actual local-forest serialization/upload payload sizes plus a simulated server-to-client global-forest broadcast payload size; the broadcast is not transmitted or reused by the current single-process RF tree-pooling loop",
         "warmup":"one prediction warm-up before inference timing; SHAP uses C3's existing sample selection and explanation method"}
    (CONFIG_DIR/"contribution4_environment.json").write_text(json.dumps(env,indent=2,default=str),encoding="utf-8")
    cfg={"seed":SEED,"n_clients":N_CLIENTS,"n_rounds":N_ROUNDS,"trees_per_client":TREES_PER_CLIENT,"global_trees":GLOBAL_TREES,"personalization_trees":PERSONALIZATION_TREES,
         "dirichlet_alpha":DIRICHLET_ALPHA,"rf_params":RF_PARAMS,"knn_neighbors":KNN_NEIGHBORS,"pearson_threshold":PEARSON_THRESHOLD,"smote_random_state":SMOTE_RANDOM_STATE,
         "timing_repetitions":TIMING_REPETITIONS,"inference_repetitions":INFERENCE_REPETITIONS,"shap_sample_method":"Contribution 3 existing select_samples; 2 detailed samples total per task (clients 0 and 5)",
         "centralized_baseline":"new evaluation-only centralized RF trained on pooled federation-training rows so the data basis matches the global FL federation shards; does not replace any existing result",
         "network_latency":"NOT MEASURED"}
    (CONFIG_DIR/"contribution4_config.json").write_text(json.dumps(cfg,indent=2,default=str),encoding="utf-8")


def main():
    environment(); task_results={}
    requested = sys.argv[1:] or ["diagnosis", "severity"]
    for task in requested:
        print(f"=== Contribution 4: {task} ===",flush=True)
        ctx=build_c2_task(task)
        cent=benchmark_centralized(ctx); print("centralized done",flush=True)
        fed=benchmark_federated(ctx); print("federated done",flush=True)
        pers=benchmark_personalized(ctx,fed[0]); print("personalized done",flush=True)
        shapdf=benchmark_shap(task,ctx,fed[0],pers[0]); print("SHAP done",flush=True)
        task_results[task]={"ctx":ctx,"central":cent,"fed":fed,"pers":pers,"shap":shapdf}
    make_summaries(task_results); plot_outputs(task_results)
    # Additional exact 50/10 structural audit.
    audit=[]
    for task,r in task_results.items():
        rd=r["fed"][2]; audit.append({"task":task,"round_count":len(rd),"client_count":r["fed"][3].client_id.nunique(),"global_trees":len(r["fed"][0].estimators_),"personalization_trees":int(r["pers"][2].local_tree_count.mean()),"network_latency":"NOT MEASURED"})
    pd.DataFrame(audit).to_csv(OUT/"summary"/"configuration_audit.csv",index=False)
    print("Contribution 4 benchmark complete.")

if __name__ == "__main__": main()
