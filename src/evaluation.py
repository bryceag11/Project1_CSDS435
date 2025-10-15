# evaluation.py
import json
from typing import Any, Dict, Optional

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    serialized = {}
    for key, value in params.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def _log_classifier_results(
    writer: SummaryWriter,
    dataset_name: str,
    classifier_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    cv_results: Dict[str, np.ndarray],
    step: int,
) -> None:
    base_tag = f"{dataset_name}/{classifier_name}"
    writer.add_scalar(f"{base_tag}/accuracy_mean", metrics['accuracy'], step)
    writer.add_scalar(f"{base_tag}/accuracy_std", metrics['accuracy_std'], step)
    writer.add_scalar(f"{base_tag}/precision_mean", metrics['precision'], step)
    writer.add_scalar(f"{base_tag}/recall_mean", metrics['recall'], step)
    writer.add_scalar(f"{base_tag}/f1_mean", metrics['f1'], step)

    for metric_key, metric_label in (
        ('test_accuracy', 'fold_accuracy'),
        ('test_precision', 'fold_precision'),
        ('test_recall', 'fold_recall'),
        ('test_f1', 'fold_f1'),
    ):
        if metric_key in cv_results:
            for fold_idx, value in enumerate(cv_results[metric_key]):
                writer.add_scalar(f"{base_tag}/{metric_label}", value, fold_idx)

    params_text = json.dumps(_serialize_params(params), indent=2, sort_keys=True)
    writer.add_text(f"{base_tag}/hyperparameters", params_text, step)


def evaluate_all_models(classifiers, X, y, cv=10, writer: Optional[SummaryWriter] = None, dataset_name: str = "default"):
    """
    Evaluate all classifiers using 10-fold cross-validation
    """
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0)
    }
    
    results = []
    
    for name, clf in classifiers.items():
        print(f"\nEvaluating {name}...")
        
        try:
            cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, 
                                       return_train_score=False, n_jobs=-1)
            
            metrics = {
                'classifier': name,
                'accuracy': np.mean(cv_results['test_accuracy']),
                'precision': np.mean(cv_results['test_precision']),
                'recall': np.mean(cv_results['test_recall']),
                'f1': np.mean(cv_results['test_f1']),
                'accuracy_std': np.std(cv_results['test_accuracy'])
            }
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            
            results.append(metrics)

            if writer is not None:
                _log_classifier_results(
                    writer,
                    dataset_name,
                    name,
                    clf.get_params(deep=True),
                    metrics,
                    cv_results,
                    len(results) - 1,
                )
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'classifier': name,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'accuracy_std': 0,
                'error': str(e)
            })
    
    return results