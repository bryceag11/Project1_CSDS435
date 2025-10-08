# classifiers.py
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_classifiers():
    """
    Initialize all classifiers
    """
    classifiers = {
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            metric='euclidean'
        ),
        'Decision Tree': DecisionTreeClassifier(
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            random_state=42
        ),
        'Naive Bayes': GaussianNB(
            var_smoothing=1e-9
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42
        ),
        'AdaBoost (DT-based)': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        )
    }
    return classifiers