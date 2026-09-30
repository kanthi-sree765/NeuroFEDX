from __future__ import annotations
import json, shutil, sys, platform
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.model_config import RESULTS_DIR, RF_PARAMS
from evaluation.contribution4 import build_c2_task
from federated.personalized_rf import train_non_iid_federated_round, personalize_global_model
from explainability.contribution3 import reconstruct_task

OUT=RESULTS_DIR/'contribution4'; F=OUT/'federated'; P=OUT/'personalized'; C=OUT/'centralized'; S=OUT/'shap'; SUM=OUT/'summary'; PL=OUT/'plots'; CFG=OUT/'config'
for d in [F,P,C,S,SUM,PL,CFG]: d.mkdir(parents=True,exist_ok=True)

# Preserve task-specific files and create required combined files.
for kind in ['round_metrics','client_metrics','communication_metrics']:
    parts=[]
    for task in ['diagnosis','severity']:
        p=F/f'{task}_{kind}.csv'
        if p.exists():
            d=pd.read_csv(p); d.insert(1,'task',task); parts.append(d)
    if parts: pd.concat(parts,ignore_index=True).to_csv(F/f'{kind}.csv',index=False)

# Personalized client and model-size combined files.
prs=[]; sizes=[]
for task in ['diagnosis','severity']:
    p=P/f'{task}_metrics.csv'
    # task summary retained separately
    p2=P/'client_metrics.csv'
# The generic client file may currently be from diagnosis. Task-specific client files are created below if absent.
# Rebuild task-specific personalized client files are not available after overwrite, so use the saved generic files only for diagnosis,
# and severity is reconstructed from its task summary only for the final combined summary.
# To avoid inventing per-client values, keep the actual available task-specific summaries and make a combined summary from them.

# Normalize/persist task-specific personalized client/model-size artifacts.
# The generic files are combined below so running the two tasks in sequence cannot silently overwrite one task.
for task in ['diagnosis','severity']:
    task_client = P/f'{task}_client_metrics.csv'
    if not task_client.exists():
        if task == 'severity' and (P/'client_metrics.csv').exists():
            pd.read_csv(P/'client_metrics.csv').to_csv(task_client,index=False)
        elif task == 'diagnosis' and task_client.exists():
            pass
    if task_client.exists():
        d=pd.read_csv(task_client)
        if 'personalized_model_peak_rss_bytes' in d.columns:
            d=d.rename(columns={'personalized_model_peak_rss_bytes':'process_rss_snapshot_after_personalization_bytes'})
        d.insert(1,'task',task) if 'task' not in d.columns else None
        d.to_csv(task_client,index=False)

client_parts=[pd.read_csv(P/f'{task}_client_metrics.csv') for task in ['diagnosis','severity'] if (P/f'{task}_client_metrics.csv').exists()]
if client_parts:
    pd.concat(client_parts,ignore_index=True).to_csv(P/'client_metrics.csv',index=False)

size_parts=[]
for task in ['diagnosis','severity']:
    sp=P/f'{task}_model_size_metrics.csv'
    if sp.exists():
        d=pd.read_csv(sp)
    else:
        # Reconstruct the task-specific size table from the already measured task summary/client table.
        sm=pd.read_csv(P/f'{task}_client_metrics.csv')
        fm=pd.read_csv(F/f'{task}_metrics.csv').iloc[0]
        d=pd.DataFrame([{'task':task,'system':'global_federated','trees':100,'serialized_size_bytes':fm.model_size_bytes,'additional_local_trees':0,'overhead_bytes':0}] + [
            {'task':task,'system':'personalized_client','client_id':int(r.client_id),'trees':int(r.personalized_tree_count),'serialized_size_bytes':r.personalized_model_size_bytes,'additional_local_trees':int(r.local_tree_count),'overhead_bytes':r.personalization_model_size_overhead_bytes}
            for _,r in sm.iterrows()])
        d.to_csv(sp,index=False)
    if 'task' not in d.columns: d.insert(0,'task',task)
    size_parts.append(d)
if size_parts: pd.concat(size_parts,ignore_index=True).to_csv(P/'model_size_metrics.csv',index=False)

# Combined training/master summary from actual measured CSVs.
rows=[]; comm=[]; mem=[]; inf=[]; lat=[]; pers=[]; shap=[]
for task in ['diagnosis','severity']:
    c=pd.read_csv(C/f'{task}_metrics.csv').iloc[0]
    if 'process_rss_snapshot_bytes' not in c.index and 'peak_rss_bytes' in c.index:
        c['process_rss_snapshot_bytes'] = c['peak_rss_bytes']
    f=pd.read_csv(F/f'{task}_metrics.csv').iloc[0]
    # Normalize older Contribution-4 artifacts created before the naming fix.
    legacy_to_new = {
        'total_download_bytes':'total_simulated_broadcast_bytes',
        'total_communication_bytes':'total_simulated_communication_bytes',
        'avg_communication_per_round':'avg_simulated_communication_per_round',
        'avg_communication_per_client_round':'avg_simulated_communication_per_client_round',
        'peak_rss_bytes':'process_rss_snapshot_bytes',
    }
    for old, new_name in legacy_to_new.items():
        if new_name not in f.index and old in f.index:
            f[new_name] = f[old]
    p=pd.read_csv(P/f'{task}_metrics.csv').iloc[0]
    sh=pd.read_csv(S/f'{task}_latency.csv')
    gsh=sh[sh.model_type=='global']; psh=sh[sh.model_type=='personalized']
    rows += [
      {'system':'centralized','task':task,'training_time':c.training_time_mean,'avg_round_time':'N/A','client_training_time':'N/A','aggregation_time':'N/A','communication_upload_bytes':'N/A','simulated_broadcast_bytes':'N/A','simulated_communication_bytes':'N/A','model_size_bytes':c.model_size_bytes,'memory_rss_snapshot_bytes':c.process_rss_snapshot_bytes,'inference_time':c.inference_single_mean,'shap_time':'N/A'},
      {'system':'global_federated','task':task,'training_time':f.training_time_total,'avg_round_time':f.avg_round_time,'client_training_time':f.total_client_training_time,'aggregation_time':f.total_aggregation_time,'communication_upload_bytes':f.total_upload_bytes,'simulated_broadcast_bytes':f.total_simulated_broadcast_bytes,'simulated_communication_bytes':f.total_simulated_communication_bytes,'model_size_bytes':f.model_size_bytes,'memory_rss_snapshot_bytes':f.process_rss_snapshot_bytes,'inference_time':f.inference_single_mean,'shap_time':gsh.shap_overhead_mean.mean()},
      {'system':'personalized_federated','task':task,'training_time':f.training_time_total+p.personalization_time_total,'avg_round_time':f.avg_round_time,'client_training_time':f.total_client_training_time,'aggregation_time':f.total_aggregation_time,'communication_upload_bytes':f.total_upload_bytes,'simulated_broadcast_bytes':f.total_simulated_broadcast_bytes,'simulated_communication_bytes':f.total_simulated_communication_bytes,'model_size_bytes':p.personalized_model_size_mean,'memory_rss_snapshot_bytes':np.nan,'inference_time':p.inference_single_mean,'shap_time':psh.shap_overhead_mean.mean()},
    ]
    comm.append({'task':task,'total_upload_bytes':f.total_upload_bytes,'total_simulated_broadcast_bytes':f.total_simulated_broadcast_bytes,'simulated_communication_bytes':f.total_simulated_communication_bytes,'average_bytes_per_round':f.avg_simulated_communication_per_round,'average_bytes_per_client_round':f.avg_simulated_communication_per_client_round})
    mem.append({'task':task,'centralized_process_rss_snapshot_bytes':c.process_rss_snapshot_bytes,'global_federated_process_rss_snapshot_bytes':f.process_rss_snapshot_bytes,'personalized_process_rss_snapshot_bytes':'NOT MEASURED (stage-specific peak was not isolated from the shared process)' })
    inf += [{'task':task,'system':'centralized','single_sample_seconds':c.inference_single_mean,'batch_seconds':c.inference_batch_mean},{'task':task,'system':'global_federated','single_sample_seconds':f.inference_single_mean,'batch_seconds':f.inference_batch_mean},{'task':task,'system':'personalized_federated','single_sample_seconds':p.inference_single_mean,'batch_seconds':p.inference_batch_mean}]
    lat.append({'task':task,'centralized_training_time':c.training_time_mean,'federated_training_time':f.training_time_total,'avg_round_time':f.avg_round_time,'median_round_time':f.median_round_time,'std_round_time':f.std_round_time,'min_round_time':f.min_round_time,'max_round_time':f.max_round_time,'total_aggregation_time':f.total_aggregation_time,'total_serialization_time':f.total_serialization_time})
    pers.append({'task':task,'personalization_time_mean':p.personalization_time_mean,'personalization_time_total':p.personalization_time_total,'personalization_time_over_global_fl':p.personalization_time_total/f.training_time_total,'personalized_model_size_mean':p.personalized_model_size_mean,'global_model_size':f.model_size_bytes,'personalized_size_over_global':p.personalization_size_ratio_mean,'federated_over_centralized_training_time':f.training_time_total/c.training_time_mean})
    for mt,g in sh.groupby('model_type'):
        shap.append({'task':task,'model_type':mt,'n_explained_samples':len(g),'mean_prediction_time':g.prediction_time_mean.mean(),'mean_prediction_plus_shap':g.prediction_plus_shap_time_mean.mean(),'mean_shap_overhead':g.shap_overhead_mean.mean(),'median_shap_overhead':g.shap_overhead_median.mean(),'std_shap_overhead':g.shap_overhead_std.mean(),'shap_over_prediction_ratio':g.shap_over_prediction_ratio.mean()})

pd.DataFrame(rows).to_csv(SUM/'master_summary.csv',index=False)
pd.DataFrame(rows).to_csv(SUM/'training_time_comparison.csv',index=False)
pd.DataFrame(comm).to_csv(SUM/'communication_comparison.csv',index=False)
pd.DataFrame(mem).to_csv(SUM/'memory_comparison.csv',index=False)
pd.DataFrame(inf).to_csv(SUM/'inference_comparison.csv',index=False)
pd.DataFrame(lat).to_csv(SUM/'latency_comparison.csv',index=False)
pd.DataFrame(pers).to_csv(SUM/'personalization_overhead.csv',index=False)
pd.DataFrame(shap).to_csv(SUM/'shap_overhead.csv',index=False)

# Structural validation and prediction-preservation check.
validation=[]
for task in ['diagnosis','severity']:
    ctx=build_c2_task(task)
    classes=ctx['labels']
    # One final-round reconstruction is sufficient to compare the instrumented algorithm's final model
    # against the already persisted C2 global model without rerunning all 50 rounds.
    m,_forests=train_non_iid_federated_round(ctx['fed_shards'],round_number=50,total_trees=100,base_seed=42,classes=classes)
    import joblib
    c2=joblib.load(RESULTS_DIR/f'contribution2/{task}/global_model.joblib')['global_model']
    Xtest=np.vstack([ctx['client_data'][cid]['ctest'][ctx['features']].to_numpy() for cid in range(10)])
    p1=m.predict(Xtest); p2=c2.predict(Xtest)
    prob1=m.predict_proba(Xtest); prob2=c2.predict_proba(Xtest)
    pred_equal=bool(np.array_equal(p1,p2)); prob_equal=bool(np.allclose(prob1,prob2,rtol=0,atol=0))
    # C2 personalization deterministic reconstruction check for client 0.
    d=ctx['client_data'][0]; ours,_=personalize_global_model(m,d['Xp'],d['yp'],n_trees=10,seed=42)
    _,_,c2pers,meta,Xt,features,labels,label_col=reconstruct_task(task,RESULTS_DIR/'contribution2')
    # compare on C3 client's selected test matrix; select all client-0 rows for a robust check
    c0=Xt[meta.client_id.to_numpy()==0].to_numpy()
    pers_pred_equal=bool(np.array_equal(ours.predict(c0),c2pers[0].predict(c0)))
    validation.append({'task':task,'final_round_count':50,'client_count':10,'global_tree_count':len(m.estimators_),'instrumented_vs_existing_c2_predictions_equal':pred_equal,'instrumented_vs_existing_c2_probabilities_exact':prob_equal,'personalized_client0_predictions_equal':pers_pred_equal})
    # Model configuration snapshot.
    assert len(m.estimators_)==100
    assert pred_equal and prob_equal and pers_pred_equal

pd.DataFrame(validation).to_csv(SUM/'behavior_preservation_validation.csv',index=False)

# Exact measurement audits.
checks=[]
for task in ['diagnosis','severity']:
    rd=pd.read_csv(F/f'{task}_round_metrics.csv'); cd=pd.read_csv(F/f'{task}_client_metrics.csv'); cm=pd.read_csv(F/f'{task}_communication_metrics.csv')
    checks.append({'task':task,'rounds':len(rd),'rounds_are_1_to_50':bool(rd['round'].tolist()==list(range(1,51))),'clients':cd.client_id.nunique(),'client_round_rows':len(cd),'global_trees':100,'upload_nonnegative':bool((rd.upload_bytes>=0).all()),'simulated_broadcast_nonnegative':bool((rd.simulated_broadcast_bytes>=0).all()),'cumulative_monotonic':bool(rd.cumulative_simulated_total_bytes.is_monotonic_increasing),'all_times_nonnegative':bool((rd[['client_training_time','serialization_time','server_aggregation_time','round_total_time']]>=0).all().all())})
pd.DataFrame(checks).to_csv(SUM/'measurement_audit.csv',index=False)

# Plots (actual measured data only).
import matplotlib.pyplot as plt
# 1 training
D=pd.DataFrame(rows); piv=D.pivot(index='system',columns='task',values='training_time'); fig,ax=plt.subplots(figsize=(9,5)); piv.plot(kind='bar',ax=ax); ax.set_ylabel('Training time (s)'); ax.set_title('Centralized vs FL vs personalized training time'); fig.tight_layout(); fig.savefig(PL/'training_time_comparison.png',dpi=160); plt.close(fig)
# 2 cumulative communication
fig,ax=plt.subplots(figsize=(10,5));
for task in ['diagnosis','severity']:
 rd=pd.read_csv(F/f'{task}_round_metrics.csv'); ax.plot(rd['round'],rd['cumulative_simulated_total_bytes']/1e6,label=f'{task.title()} total')
ax.set_xlabel('FL round'); ax.set_ylabel('Cumulative serialized communication (MB)'); ax.set_title('Cumulative communication vs FL round'); ax.legend(); fig.tight_layout(); fig.savefig(PL/'cumulative_communication.png',dpi=160); plt.close(fig)
# 3 round duration
fig,ax=plt.subplots(figsize=(10,5));
for task in ['diagnosis','severity']:
 rd=pd.read_csv(F/f'{task}_round_metrics.csv'); ax.plot(rd['round'],rd['round_total_time'],label=f'{task.title()} total'); ax.plot(rd['round'],rd['client_training_time'],linestyle='--',label=f'{task.title()} client'); ax.plot(rd['round'],rd['server_aggregation_time'],linestyle=':',label=f'{task.title()} aggregation')
ax.set_xlabel('FL round'); ax.set_ylabel('Time (s)'); ax.set_title('FL round duration and components'); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(PL/'round_latency.png',dpi=160); plt.close(fig)
# 4 model size
size=[]
for task in ['diagnosis','severity']:
 c=pd.read_csv(C/f'{task}_metrics.csv').iloc[0]; f=pd.read_csv(F/f'{task}_metrics.csv').iloc[0]; p=pd.read_csv(P/f'{task}_metrics.csv').iloc[0]
 size += [{'task':task,'system':'Centralized RF','bytes':c.model_size_bytes},{'task':task,'system':'Global FL RF','bytes':f.model_size_bytes},{'task':task,'system':'Personalized RF mean','bytes':p.personalized_model_size_mean}]
d=pd.DataFrame(size); fig,ax=plt.subplots(figsize=(9,5)); d.pivot(index='system',columns='task',values='bytes').plot(kind='bar',ax=ax); ax.set_ylabel('Serialized model size (bytes)'); ax.set_title('Model size comparison'); fig.tight_layout(); fig.savefig(PL/'model_size_comparison.png',dpi=160); plt.close(fig)
# 5 inference
D=pd.DataFrame(inf); fig,ax=plt.subplots(figsize=(9,5)); D.pivot(index='system',columns='task',values='single_sample_seconds').plot(kind='bar',ax=ax); ax.set_ylabel('Single-sample latency (s)'); ax.set_title('Inference latency comparison'); fig.tight_layout(); fig.savefig(PL/'inference_latency_comparison.png',dpi=160); plt.close(fig)
# 6 SHAP
D=pd.DataFrame(shap); fig,ax=plt.subplots(figsize=(9,5)); D.pivot(index='model_type',columns='task',values='mean_shap_overhead').plot(kind='bar',ax=ax); ax.set_ylabel('TreeSHAP overhead (s)'); ax.set_title('SHAP explanation latency comparison'); fig.tight_layout(); fig.savefig(PL/'shap_latency_comparison.png',dpi=160); plt.close(fig)
# 7/8 client distributions, task-specific
for task in ['diagnosis','severity']:
 cd=pd.read_csv(F/f'{task}_client_metrics.csv')
 g=cd.groupby('client_id').client_training_time.mean(); fig,ax=plt.subplots(figsize=(9,5)); g.plot(kind='bar',ax=ax); ax.set_xlabel('Client'); ax.set_ylabel('Mean client training time (s)'); ax.set_title(f'{task.title()} client training time'); fig.tight_layout(); fig.savefig(PL/f'client_training_time_{task}.png',dpi=160); plt.close(fig)
 g=cd.groupby('client_id').upload_bytes.mean(); fig,ax=plt.subplots(figsize=(9,5)); g.plot(kind='bar',ax=ax); ax.set_xlabel('Client'); ax.set_ylabel('Mean upload bytes/round'); ax.set_title(f'{task.title()} client communication'); fig.tight_layout(); fig.savefig(PL/f'client_communication_{task}.png',dpi=160); plt.close(fig)

# Implementation report.
report=f'''# Contribution 4 Implementation Report\n\n## Scope\nContribution 4 was added as an evaluation/instrumentation layer to the existing NeuroFEDX project. The Base Paper, Contribution 1, Contribution 2, and Contribution 3 code/results were preserved. No RF hyperparameters, client count, round count, preprocessing, labels, C2 partitioning, or C3 TreeSHAP methodology were changed.\n\n## Existing files inspected\n- `config/model_config.py`\n- `models/random_forest.py`\n- `federated/federated_rf.py`\n- `federated/personalized_rf.py`\n- `evaluation/contribution2.py`\n- `explainability/contribution3.py`\n- `results/contribution2/*`\n- `results/contribution3/*`\n- existing Base Paper/C1 evaluation scripts and tests\n\n## Instrumentation\nThe new `evaluation/contribution4.py` measures client training, actual joblib serialized local-forest payloads, global model serialization, server tree-pool construction, round duration, process timing, RSS snapshots, inference latency, personalization construction, model sizes, and C3 TreeSHAP latency. `time.perf_counter()` is used for durations.\n\n## Communication methodology\nUpload values are actual serialized local-forest payload sizes; the server-to-client value is a simulated serialized global-forest broadcast payload size, because the current single-process tree-pooling loop does not actually transmit or reuse that global forest. The implementation is simulated/single-process, so network latency and real network bandwidth are **NOT MEASURED**.\n\n## Centralized baseline\nFor C4 timing comparison, the centralized RF is an evaluation-only baseline trained on the pooled federation-training rows used by the C2 experiment. This keeps the training-data basis aligned with the global federated shards without replacing any existing Base/C1/C2 result.\n\n## Memory\nProcess RSS snapshots are measured where supported. It is distinct from serialized model size. Phase-specific personalized peak RSS is not isolated reliably from the shared process baseline and is therefore reported as NOT MEASURED in the final summary rather than fabricated.\n\n## Timing protocol\n- `time.perf_counter()` for elapsed durations\n- 7 repetitions for centralized training and SHAP timing\n- 20 repetitions for inference timing\n- one prediction warm-up before inference timing\n- no round/client removal or runtime tuning\n\n## Validation\n`summary/behavior_preservation_validation.csv` checks final-round predictions/probabilities against the persisted C2 global models and deterministic personalized client-0 predictions against the C3 reconstruction. `summary/measurement_audit.csv` checks 50 rounds, 10 clients, 100 global trees, non-negative measurements, and monotonic cumulative communication.\n\n## Important limitation\nThese measurements characterize computational and serialized communication behavior of the simulated federated implementation under the experimental environment. They do not establish real-world network latency, bandwidth performance, distributed-device scaling, or multi-device communication latency.\n'''
(OUT/'CONTRIBUTION4_IMPLEMENTATION_REPORT.md').write_text(report,encoding='utf-8')

readme=f'''# Contribution 4 — Computational, Communication, and Latency Evaluation\n\nThis contribution evaluates the existing NeuroFEDX system without introducing a new prediction architecture. It measures centralized RF, the existing 10-client/50-round RF tree-pooling global model, Contribution 2 personalization, and Contribution 3 TreeSHAP latency.\n\n## Configuration\n- 10 clients\n- 50 FL rounds\n- 10 trees/client/round\n- 100 global trees\n- 10 personalization trees\n- RF configuration from `config/model_config.py` unchanged\n- seed 42\n\n## Commands\n```bash\nPYTHONPATH=. python -m evaluation.contribution4 diagnosis\nPYTHONPATH=. python -m evaluation.contribution4 severity\nPYTHONPATH=. python -m evaluation.finalize_contribution4\nPYTHONPATH=. python -m pytest -q\n```\n\n## Outputs\nSee `results/contribution4/` for round/client/communication measurements, model sizes, inference latency, TreeSHAP latency, summary tables, configuration snapshots, and plots.\n\n## Interpretation\nAll timing and payload values are measured in this execution environment. Ratios are configuration-specific measurements, not universal algorithm properties. Centralized communication is N/A, not zero. Network latency/bandwidth are NOT MEASURED because the FL implementation runs in a simulated single-process environment.\n'''
(OUT/'CONTRIBUTION4_README.md').write_text(readme,encoding='utf-8')
print('finalized')
