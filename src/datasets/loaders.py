"""Data loaders for Adult Income and German Credit datasets."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def load_adult():
    """UCI Adult Income (id=2).

    Sensitive : sex  (Male → s=1, Female → s=0).
    Target    : income >50 K → y=1.

    Returns
    -------
    X : ndarray, shape (n, d)   — standardised features (sex removed).
    y : ndarray, shape (n,)     — binary target.
    s : ndarray, shape (n,)     — binary sensitive attribute.
    """
    ds = fetch_ucirepo(id=2)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1).dropna()
    target_col = ds.data.targets.columns[0]

    df[target_col] = df[target_col].astype(str).str.strip().str.rstrip(".")
    y = (df[target_col] == ">50K").astype(int).values
    s = (df["sex"].astype(str).str.strip() == "Male").astype(int).values

    feat_df = df.drop(columns=[target_col, "sex"])
    feat_df = pd.get_dummies(feat_df, drop_first=True)
    X = StandardScaler().fit_transform(feat_df.values.astype(float))
    return X, y, s


def load_german():
    """UCI Statlog German Credit (id=144).

    Sensitive : sex derived from personal-status column
                (A91/A93/A94 → Male → s=1; A92/A95 → Female → s=0).
    Target    : credit risk  (1 = good → y=1, 2 = bad → y=0 in raw data).

    Returns
    -------
    X, y, s  — same convention as :func:`load_adult`.
    """
    ds = fetch_ucirepo(id=144)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1).dropna()
    target_col = ds.data.targets.columns[0]

    # Locate personal-status column (values starting with 'A9')
    pstatus_col = None
    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).str.startswith("A9").any():
            pstatus_col = c
            break
    assert pstatus_col is not None, "Could not locate personal-status column"

    male_codes = {"A91", "A93", "A94"}
    s = df[pstatus_col].astype(str).str.strip().isin(male_codes).astype(int).values
    y = (df[target_col].astype(int) == 1).astype(int).values

    feat_df = df.drop(columns=[target_col, pstatus_col])
    feat_df = pd.get_dummies(feat_df, drop_first=True)
    X = StandardScaler().fit_transform(feat_df.values.astype(float))
    return X, y, s
