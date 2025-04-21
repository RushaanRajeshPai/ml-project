# utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drops irrelevant columns and handles missing values."""
    df = df.drop(columns=[
        'row_id', 'bank_interest_rate', 'mm_interest_rate',
        'mfi_interest_rate', 'other_fsp_interest_rate'
    ])
    df = df.dropna(subset=['education_level', 'share_hh_income_provided'])
    return df

def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encodes all object columns."""
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def generate_target(df: pd.DataFrame, country_col='country') -> pd.DataFrame:
    """Creates a binary target based on country frequency."""
    country_counts = df[country_col].value_counts()
    poverty_countries = country_counts[country_counts < country_counts.median()].index.tolist()
    df['is_poverty_country'] = df[country_col].apply(lambda x: 1 if x in poverty_countries else 0)
    return df
