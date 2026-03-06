
# Data manipulation
import pandas as pd

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Model saving
import joblib

# Hugging Face
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow
import mlflow


# --------------------------------------------------
# MLflow Setup
# --------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-experiment")

api = HfApi()


# --------------------------------------------------
# Load Data from Hugging Face Dataset
# --------------------------------------------------

Xtrain_path = "hf://datasets/maha16694-svg/tourism-package/Xtrain.csv"
Xtest_path = "hf://datasets/maha16694-svg/tourism-package/Xtest.csv"
ytrain_path = "hf://datasets/maha16694-svg/tourism-package/ytrain.csv"
ytest_path = "hf://datasets/maha16694-svg/tourism-package/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# --------------------------------------------------
# Feature Lists
# --------------------------------------------------

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


# --------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)


# --------------------------------------------------
# Base Model
# --------------------------------------------------

rf_model = RandomForestClassifier(random_state=42)


# --------------------------------------------------
# Hyperparameter Grid
# --------------------------------------------------

param_grid = {

    'randomforestclassifier__n_estimators': [100, 150],

    'randomforestclassifier__max_depth': [5, 10],

    'randomforestclassifier__min_samples_split': [2, 5]

}


# --------------------------------------------------
# Model Pipeline
# --------------------------------------------------

model_pipeline = make_pipeline(preprocessor, rf_model)


# --------------------------------------------------
# MLflow Training
# --------------------------------------------------

with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain.values.ravel())

    results = grid_search.cv_results_

    # Log all parameter combinations
    for i in range(len(results['params'])):

        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

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


# --------------------------------------------------
# Save Model
# --------------------------------------------------

model_path = "tourism_package_model.joblib"

joblib.dump(best_model, model_path)

mlflow.log_artifact(model_path, artifact_path="model")

print("Model saved:", model_path)


# --------------------------------------------------
# Upload Model to Hugging Face
# --------------------------------------------------

repo_id = "maha16694-svg/tourism-package-model"

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

print("Model uploaded to Hugging Face")
