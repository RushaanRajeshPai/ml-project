# ppi_prediction_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# === Load Data ===
file_path = 'data/survey_data.csv'
df = pd.read_csv(file_path)

# === Clean Data ===
df_clean = df.drop(columns=[
    'row_id', 'bank_interest_rate', 'mm_interest_rate',
    'mfi_interest_rate', 'other_fsp_interest_rate'
])
df_clean = df_clean.dropna(subset=['education_level', 'share_hh_income_provided'])

# === Encode Categorical Columns ===
label_encoders = {}
for col in df_clean.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# === Define Target for Training ===
country_counts = df_clean['country'].value_counts()
poverty_countries = country_counts[country_counts < country_counts.median()].index.tolist()
df_clean['is_poverty_country'] = df_clean['country'].apply(lambda x: 1 if x in poverty_countries else 0)

# === Define Features & Target ===
X = df_clean.drop(columns=['country', 'is_poverty_country'])
y = df_clean['is_poverty_country']

# === Train Random Forest Model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Predict PPI (Probability of Poverty) ===
df_clean['predicted_ppi'] = clf.predict_proba(X)[:, 1]

# === Calculate Country-wise Average PPI ===
country_ppi = df_clean.groupby('country')['predicted_ppi'].mean().reset_index()
country_ppi['poverty_label'] = country_ppi['predicted_ppi'].apply(lambda x: 'POOR' if x >= 0.5 else 'NOT POOR')

# === Map Country Numbers Back to Labels ===
reverse_country_map = dict(enumerate(label_encoders['country'].classes_))
country_ppi['country_label'] = country_ppi['country'].map(reverse_country_map)

# === Reorder Columns and Save ===
country_ppi = country_ppi[['country', 'country_label', 'predicted_ppi', 'poverty_label']]
os.makedirs('outputs', exist_ok=True)
country_ppi.to_csv('outputs/country_ppi_results.csv', index=False)

print("\nCountry-wise PPI Calculation Complete. Results saved to /outputs/country_ppi_results.csv")
