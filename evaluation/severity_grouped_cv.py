"""Contribution 1: participant-level 5-fold CV for CDR severity."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.model_config import RESULTS_DIR, KNN_NEIGHBORS, SMOTE_RANDOM_STATE, PEARSON_THRESHOLD
from preprocessing.preprocess_common import pearson_prune, smote_resample
from models.random_forest import train_rf
from evaluation.severity_prediction import build_severity_dataframe, metric_summary

N_FOLDS = 5

def main():
    df, feature_cols = build_severity_dataframe()
    gkf = GroupKFold(n_splits=N_FOLDS)
    summaries=[]; per=[]; checks=[]
    for fold,(tr,va) in enumerate(gkf.split(df, groups=df.OASISID)):
        train=df.iloc[tr].reset_index(drop=True); val=df.iloc[va].reset_index(drop=True)
        overlap=set(train.OASISID)&set(val.OASISID)
        assert not overlap
        checks.append({'fold':fold,'train_subjects':train.OASISID.nunique(),'validation_subjects':val.OASISID.nunique(),'overlap':len(overlap)})
        imp=KNNImputer(n_neighbors=KNN_NEIGHBORS)
        Xt=imp.fit_transform(train[feature_cols]); Xv=imp.transform(val[feature_cols])
        Xt=pd.DataFrame(Xt,columns=feature_cols); Xv=pd.DataFrame(Xv,columns=feature_cols)
        keep,drop=pearson_prune(Xt,feature_cols,threshold=PEARSON_THRESHOLD,verbose=False)
        Xtr,ytr=smote_resample(Xt[keep].to_numpy(),train.severity_label.to_numpy(),k_neighbors=5,random_state=SMOTE_RANDOM_STATE+fold)
        model=train_rf(Xtr,ytr)
        summary,pc,cm,_,_=metric_summary(model,Xv[keep].to_numpy(),val.severity_label.to_numpy(),f'severity_grouped_cv_fold{fold}')
        summary['fold']=fold; summary['n_features_after_pearson']=len(keep); summaries.append(summary)
        pc['fold']=fold; per.append(pc)
        print(f'Fold {fold}: train_subj={train.OASISID.nunique()} val_subj={val.OASISID.nunique()} overlap=0 | acc={summary["accuracy"]:.4f} bal_acc={summary["balanced_accuracy"]:.4f} macro_f1={summary["macro_f1"]:.4f}')
    r=pd.DataFrame(summaries); p=pd.concat(per,ignore_index=True); c=pd.DataFrame(checks)
    r.to_csv(RESULTS_DIR/'severity_grouped_cv_results.csv',index=False)
    p.to_csv(RESULTS_DIR/'severity_grouped_cv_per_class.csv',index=False)
    c.to_csv(RESULTS_DIR/'severity_grouped_cv_subject_overlap_check.csv',index=False)
    metrics=['accuracy','balanced_accuracy','macro_precision','macro_recall','macro_f1','weighted_f1','auc_macro_ovr']
    rows=[]
    for m in metrics: rows.append({'metric':m,'mean':r[m].mean(),'std':r[m].std()})
    pd.DataFrame(rows).to_csv(RESULTS_DIR/'severity_grouped_cv_summary.csv',index=False)
    print('\nSeverity grouped 5-fold mean +/- std:')
    for row in rows: print(f"{row['metric']:20s}: {row['mean']:.4f} +/- {row['std']:.4f}")

if __name__=='__main__': main()
