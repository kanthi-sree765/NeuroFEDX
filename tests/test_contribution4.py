from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'contribution4'


def test_required_files_nonempty():
    files=[
      OUT/'centralized/diagnosis_metrics.csv',OUT/'centralized/severity_metrics.csv',
      OUT/'federated/round_metrics.csv',OUT/'federated/client_metrics.csv',OUT/'federated/communication_metrics.csv',
      OUT/'federated/diagnosis_metrics.csv',OUT/'federated/severity_metrics.csv',
      OUT/'personalized/client_metrics.csv',OUT/'personalized/model_size_metrics.csv',
      OUT/'personalized/diagnosis_metrics.csv',OUT/'personalized/severity_metrics.csv',
      OUT/'shap/diagnosis_latency.csv',OUT/'shap/severity_latency.csv',
      OUT/'summary/training_time_comparison.csv',OUT/'summary/communication_comparison.csv',OUT/'summary/memory_comparison.csv',
      OUT/'summary/inference_comparison.csv',OUT/'summary/latency_comparison.csv',OUT/'summary/personalization_overhead.csv',OUT/'summary/shap_overhead.csv',OUT/'summary/master_summary.csv',
      OUT/'config/contribution4_environment.json',OUT/'config/contribution4_config.json',
      OUT/'CONTRIBUTION4_README.md',OUT/'CONTRIBUTION4_IMPLEMENTATION_REPORT.md'
    ]
    assert all(p.exists() and p.stat().st_size>0 for p in files)


def test_50_rounds_10_clients_and_monotonic_communication():
    d=pd.read_csv(OUT/'federated/round_metrics.csv')
    assert set(d.task)=={'diagnosis','severity'}
    for task,g in d.groupby('task'):
        assert len(g)==50
        assert g['round'].tolist()==list(range(1,51))
        assert (g['upload_bytes']>=0).all() and (g['simulated_broadcast_bytes']>=0).all()
        assert g['cumulative_simulated_total_bytes'].is_monotonic_increasing
        assert np.isclose((g.upload_bytes+g.simulated_broadcast_bytes-g.simulated_total_communication).abs().max(),0)
    c=pd.read_csv(OUT/'federated/client_metrics.csv')
    assert c.client_id.nunique()==10
    assert len(c)==1000


def test_na_and_positive_timings():
    c=pd.read_csv(OUT/'summary/master_summary.csv', keep_default_na=False)
    central=c[c.system=='centralized']
    assert (central.communication_upload_bytes=='N/A').all()
    assert (central.simulated_broadcast_bytes=='N/A').all()
    assert (central.simulated_communication_bytes=='N/A').all()
    for col in ['training_time','model_size_bytes','inference_time']:
        vals=c[c[col].map(lambda x: x!='N/A')][col].astype(float)
        assert (vals>=0).all()


def test_structure_and_validation():
    a=pd.read_csv(OUT/'summary/behavior_preservation_validation.csv')
    assert len(a)==2
    assert (a.final_round_count==50).all()
    assert (a.client_count==10).all()
    assert (a.global_tree_count==100).all()
    assert a.instrumented_vs_existing_c2_predictions_equal.all()
    assert a.instrumented_vs_existing_c2_probabilities_exact.all()
    assert a.personalized_client0_predictions_equal.all()


def test_plots_and_config():
    names=['cumulative_communication.png','round_latency.png','training_time_comparison.png','model_size_comparison.png','inference_latency_comparison.png','shap_latency_comparison.png']
    for n in names: assert (OUT/'plots'/n).exists() and (OUT/'plots'/n).stat().st_size>0
    cfg=pd.read_json(OUT/'config/contribution4_config.json',typ='series')
    assert int(cfg['n_clients'])==10 and int(cfg['n_rounds'])==50 and int(cfg['global_trees'])==100 and int(cfg['personalization_trees'])==10
