"""TreeSHAP explanation for Contribution 1 severity models."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR
from explainability.shap_analysis import balanced_sample, normalize_shap_values, explain_model


def main():
    import shap
    bundle=joblib.load(RESULTS_DIR/'severity_models.joblib')
    X=np.asarray(bundle['X_test']); y=np.asarray(bundle['y_test']); cols=list(bundle['feature_cols'])
    picks=balanced_sample(y)
    Xdf=pd.DataFrame(X,columns=cols).iloc[picks].reset_index(drop=True)
    class_names=[0,1,2]
    rows=[]
    for name in ['centralized','federated']:
        imp=explain_model(name,bundle[name],Xdf,class_names,'severity')
        # explain_model writes generic severity SHAP artifacts and audit entries.
        imp.to_csv(RESULTS_DIR/f'severity_shap_importance_{name}.csv')
        rows.append(imp)
    print('Severity TreeSHAP completed using shap.TreeExplainer', shap.__version__)

if __name__=='__main__': main()
