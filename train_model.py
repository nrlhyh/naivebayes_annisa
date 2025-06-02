import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load data
data = pd.read_csv('data/datapenelitian_annisa.csv', sep=';')
print("Kolom:", data.columns)

# Fitur & target
X = data.drop('class', axis=1)
y = data['class']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Pipeline: Scaling + Model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

# Parameter tuning
param_grid = {
    'nb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Evaluasi hasil tuning
print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# Simpan model dan label encoder
joblib.dump(grid.best_estimator_, 'app/model/model.pkl')
joblib.dump(le, 'app/model/label_encoder.pkl')

print("Model berhasil disimpan.")