# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os

# === Load Data ===
data_path = os.path.join('data', 'survey_data.csv')
df = pd.read_csv(data_path)

# === Drop unnecessary columns ===
df_clean = df.drop(columns=[
    'row_id', 'bank_interest_rate', 'mm_interest_rate',
    'mfi_interest_rate', 'other_fsp_interest_rate'
])

# Drop rows with missing values in key columns
df_clean = df_clean.dropna(subset=['education_level', 'share_hh_income_provided'])

# === Encode Categorical Columns ===
label_encoders = {}
for col in df_clean.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

# === Define Target Variable ===
country_counts = df_clean['country'].value_counts()
poverty_countries = country_counts[country_counts < country_counts.median()].index.tolist()
df_clean['is_poverty_country'] = df_clean['country'].apply(lambda x: 1 if x in poverty_countries else 0)

# === Features and Target ===
X = df_clean.drop(columns=['country', 'is_poverty_country'])
y = df_clean['is_poverty_country']

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Predictions ===
y_pred = clf.predict(X_test)

# === Save Evaluation Report ===
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

os.makedirs('outputs', exist_ok=True)
with open(os.path.join('outputs', 'classification_report.txt'), 'w') as f:
    f.write(report)
    f.write('\nConfusion Matrix:\n')
    f.write(str(conf_matrix))

# === Save Feature Importances ===
feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importances.head(20).to_csv(os.path.join('outputs', 'feature_importance.csv'))

print("Model training complete. Reports saved to /outputs")
