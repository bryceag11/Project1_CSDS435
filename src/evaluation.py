# evaluation.py
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

def evaluate_all_models(classifiers, X, y, cv=10):
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