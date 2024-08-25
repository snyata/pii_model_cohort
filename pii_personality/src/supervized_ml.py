from .constants import PROJECT, VERSION
import json
import os
import random
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load environment variables
load_dotenv()

# Load project details


project = PROJECT
version = VERSION

def initialize_project(project: str, version: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Initializes a project and performs a train-test split on the processed data.

    Parameters:
        project (str): The name of the project.
        version (str): The version of the project.

    Returns:
        tuple: A tuple containing the following:
            - X_train (pd.DataFrame): The training features.
            - X_test (pd.DataFrame): The testing features.
            - y_train (pd.Series): The training targets.
            - y_test (pd.Series): The testing targets.
            - models (dict): A dictionary of model instances.
    """
    print('initializing project')
    data_path = "/Users/nullzero/Documents/repos/github.com/privacy-identity/vda-simulation-medical/vda-sim-medical/data/processed/PII_Customer_Personality_Analysis/data/2024_08_25_PII_Customer_Personality_Analysis_v0.1.csv"
    
    # Load the processed data
    df_processed = pd.read_csv(data_path)
    
    print("splitting dataset...")
    # Train-Test Split
    X = df_processed.drop(columns=['target'])
    y = df_processed['target']

    # Select the top 10 features
    print("Selecting best features...")
    selector = SelectKBest(score_func=f_classif, k=10)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    X = pd.DataFrame(X_new, columns=selected_features)

    # Log the selected features to W&B
    wandb.init(project=project, entity="orionai", name="supervized_binary_classification", job_type="supervized_train")
    wandb.log({"selected_features": selected_features.tolist()})

    # Normalize the data
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, probability=True)
    }
    
    return X_train, X_test, y_train, y_test, models


def training_clf(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, models: Dict[str, Any], project: str, version: str) -> Dict[str, Any]:
    """
    Trains and logs multiple classification models using Weights & Biases (W&B).

    Args:
        X_train (pd.DataFrame): The training features.
        X_test (pd.DataFrame): The testing features.
        y_train (pd.Series): The training targets.
        y_test (pd.Series): The testing targets.
        models (dict): A dictionary of classification models to train and log.
        project (str): The W&B project name.
        version (str): The model version.

    Returns:
        dict: A dictionary containing the model name, classification report, confusion matrix, accuracy, ROC AUC, and F1 score for each model.
    """
    results = {}

    for model_name, model in models.items():
        # Initialize a new W&B run for each model
        run = wandb.init(project=project, entity="orionai", job_type="supervized_train", name=model_name)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        f1_metric = f1_score(y_test, y_pred)
        
        # Log metrics
        wandb.log({
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "f1_score": f1_metric
        })
        
        # Log model
        wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, y_pred, y_prob, labels=["Not Buy", "Buy"])
        
        # Save the model to a file
        model_filename = f"{model_name.replace(' ', '_').lower()}_model_v{version}.pkl"
        torch.save(model, model_filename)
        
        # Create and log the W&B artifact for the model
        model_artifact = wandb.Artifact(name=f"{model_name.replace(' ', '_').lower()}_v{version}", type='model')
        model_artifact.add_file(model_filename)
        wandb.log_artifact(model_artifact)
        
        # Log classification report and confusion matrix
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        wandb.log({
            "classification_report": class_report,
            "confusion_matrix": conf_matrix
        })
        
        results[model_name] = {
            "clf_report": class_report,
            "conf_matrix": conf_matrix,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "f1_score": f1_metric
        }
        
        # End W&B run for this model
        run.finish()

    return results


def json_convert(input_dict: Dict[str, Any], project: str) -> str:
    """
    Converts a dictionary into a JSON file and saves it to a specified directory.

    Args:
        input_dict (dict): The dictionary to be converted into a JSON file.
        project (str): The name of the project for directory organization.

    Returns:
        str: The file path where the JSON file is saved.
    """
    # Ensure the folder exists
    folder_path = f"../data/{project}/results/"
    os.makedirs(folder_path, exist_ok=True)

    file_name = f"{project}_supervized_v{random.randint(1, 100)}.json"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'w') as json_file:
        json.dump(input_dict, json_file, indent=4)
    
    print(f"Results saved to {file_path}")
    
    return file_path


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Initializing project...")
    X_train, X_test, y_train, y_test, models = initialize_project(project, version)
    
    print("Training classifiers...")
    clf_train_results = training_clf(X_train, X_test, y_train, y_test, models, project, version)
    
    print("Saving results to JSON...")
    json_convert(clf_train_results, project)
    
    print("Finished.")


if __name__ == '__main__':
    main()

