# ==============================================
# Tourism Package Prediction - Full Training Code
# ==============================================
# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

import joblib

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

import mlflow

# -----------------------------
# 2. MLflow Setup
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-experiment")

api = HfApi()

# -----------------------------
# 3. Load Dataset from Hugging Face
# -----------------------------

data_path = "hf://datasets/maha16694/tourism-package/tourism.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully")
print("Shape:", df.shape)

# -----------------------------
# 4. Split Features and Target
# -----------------------------

X = df.drop("Type", axis=1)
y = df["Type"]

# -----------------------------
# 5. Train Test Split
# -----------------------------

Xtrain, Xtest, ytrain, ytest = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# -----------------------------
# 6. Feature Lists
# -----------------------------

numeric_features = [
'Age',
'NumberOfPersonVisiting',
'PreferredPropertyStar',
'NumberOfTrips',
'PitchSatisfactionScore',
'NumberOfFollowups',
'DurationOfPitch',
'MonthlyIncome'
]

categorical_features = [
'TypeofContact',
'Occupation',
'Gender',
'MaritalStatus',
'ProductPitched',
'Designation'
]

# -----------------------------
# 7. Preprocessing Pipeline
# -----------------------------

preprocessor = make_column_transformer(
(StandardScaler(), numeric_features),
(OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# -----------------------------
# 8. Model
# -----------------------------

rf_model = RandomForestClassifier(random_state=42)

# -----------------------------
# 9. Hyperparameter Grid
# -----------------------------

param_grid = {
'randomforestclassifier__n_estimators': [100, 150],
'randomforestclassifier__max_depth': [5, 10],
'randomforestclassifier__min_samples_split': [2, 5]
}

# -----------------------------
# 10. Pipeline
# -----------------------------

model_pipeline = make_pipeline(preprocessor, rf_model)

# -----------------------------
# 11. Training with MLflow
# -----------------------------

with mlflow.start_run():

```
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_

# Log best parameters
mlflow.log_params(grid_search.best_params_)

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

train_report = classification_report(ytrain, y_pred_train, output_dict=True)
test_report = classification_report(ytest, y_pred_test, output_dict=True)

# Log metrics
mlflow.log_metrics({
    "train_accuracy": train_report['accuracy'],
    "test_accuracy": test_report['accuracy']
})

print("Train Accuracy:", train_report['accuracy'])
print("Test Accuracy:", test_report['accuracy'])
```

# -----------------------------
# 12. Save Model
# -----------------------------

model_path = "tourism_package_model.joblib"

joblib.dump(best_model, model_path)

mlflow.log_artifact(model_path)

print("Model saved:", model_path)

# -----------------------------
# 13. Upload Model to Hugging Face
# -----------------------------

repo_id = "maha16694/tourism-package-model"

try:
api.repo_info(repo_id=repo_id, repo_type="model")
print("Model repo exists")

except RepositoryNotFoundError:
create_repo(repo_id=repo_id, repo_type="model", private=False)
print("Model repo created")

api.upload_file(
path_or_fileobj=model_path,
path_in_repo=model_path,
repo_id=repo_id,
repo_type="model"
)

print("Model uploaded to Hugging Face successfully")
