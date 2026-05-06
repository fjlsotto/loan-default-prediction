import pandas as pd
import numpy as np


MISSING_THRESHOLD = 0.60


def drop_high_missing(df: pd.DataFrame, threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    return df.drop(columns=cols_to_drop)


def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', np.nan)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365
    df['ID_PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, 1)
    df['EMPLOYMENT_TO_AGE'] = df['EMPLOYMENT_YEARS'] / df['AGE_YEARS']
    return df


def add_ext_source_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    available = [c for c in ext_cols if c in df.columns]
    df['EXT_SOURCE_MEAN'] = df[available].mean(axis=1)
    df['EXT_SOURCE_MIN'] = df[available].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[available].max(axis=1)
    df['EXT_SOURCE_STD'] = df[available].std(axis=1)
    df['EXT_SOURCE_PROD'] = df[available].prod(axis=1)
    return df


def add_document_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
    if doc_cols:
        df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    binary_cols = [c for c in cat_cols if df[c].nunique() <= 2]
    multi_cols = [c for c in cat_cols if df[c].nunique() > 2]
    for col in binary_cols:
        df[col] = pd.factorize(df[col])[0]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False, dtype=int)
    return df


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns[df.isnull().any()]:
        df[col] = df[col].fillna(df[col].median())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_high_missing(df)
    df = fix_anomalies(df)
    df = add_time_features(df)
    df = add_ratio_features(df)
    df = add_ext_source_features(df)
    df = add_document_features(df)
    df = encode_categoricals(df)
    df = df.drop(columns=['AGE_GROUP'], errors='ignore')
    df = impute_median(df)
    return df
