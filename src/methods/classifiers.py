"""Scikit-learn classifier factories for baseline and base models."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def lr_factory():
    """Logistic Regression (liblinear solver)."""
    return LogisticRegression(max_iter=1000, solver="liblinear")


def rf_factory():
    """Random Forest with 200 trees."""
    return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)


def svm_factory():
    """Linear SVM (LinearSVC)."""
    return LinearSVC(C=1.0, max_iter=2000, random_state=0, dual="auto")
