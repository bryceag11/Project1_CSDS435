# classifiers.py
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

def get_classifiers():
    """
    Initialize all classifiers
    """
    classifiers = {
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), 
                                        max_iter=1000, 
                                        random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', random_state=42),
        'AdaBoost-DT': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=50,
            random_state=42
        )
    }
    return classifiers