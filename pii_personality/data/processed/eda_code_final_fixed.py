import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from dotenv import load_dotenv
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..pii_personality.src.constants import PROJECT, VERSION

load_dotenv()

project = PROJECT
version = VERSION

# Log in to W&B
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Initialize W&B run
wandb.init(project=project, entity="orionai", name="PII_customer_relationship", job_type="dataset")

if os.path.exists(f'./{project}/') is False:
    os.makedirs(f'./{project}/')
    os.makedirs(f'./{project}/artifacts/')
    os.makedirs(f'./{project}/data/')
    
print("Loading data...")
# Load the data
file_path = 'customer_profile_marketing.csv'
df = pd.read_csv(file_path)
df.rename(columns={'Response': 'target'}, inplace=True)

# Log the raw data as a W&B artifact
raw_data_artifact = wandb.Artifact('customer_profile_marketing_raw', type='dataset')
raw_data_artifact.add_file(file_path)
wandb.log_artifact(raw_data_artifact)

# Drop irrelevant columns
df = df.drop(columns=['Unnamed: 0', 'ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue'])

print("Cleaning Data...")
# Handle missing values (if any)
df_copy = df.copy()
df = df.dropna()
print("Dropped missing values...")

# Log a table with the cleaned data (before feature engineering)
wandb.log({"cleaned_data": wandb.Table(dataframe=df)})

# Feature Engineering
df['Age'] = 2024 - df['Year_Birth']  # Assuming the current year is 2024
df = df.drop(columns=['Year_Birth'])

print("Converting categorical variables...")
# Convert categorical variables to numerical format using LabelEncoder
label_encoder = LabelEncoder()
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Marital_Status'] = label_encoder.fit_transform(df['Marital_Status'])
# Log a table with the data after feature engineering
wandb.log({"feature_engineered_data": wandb.Table(dataframe=df)})


# Splitting data (train_test_split in model training files)
df_features = df.copy()

X = df_features

print("Normalising numerical values...")

# Normalize the relevant numerical features
scaler = StandardScaler()
if y.values.any():
    numerical_cols = X.select_dtypes(include=[np.number], exclude=["target"]).columns.tolist()

df_transform = scaler.fit_transform(X[numerical_cols])
df_transform = pd.DataFrame(df_transform)

# Log the processed data table
pc_artifact = wandb.Artifact("processed_data", type="dataset")
wandb.log_artifact(pc_artifact)
wandb.log({"processed_data": wandb.Table(dataframe=df_transform)})

print("beginning encoding...")
pre_encoding_path = f"./{project}/data/df_pre_encoding_{project}.csv"
# Log the processed data as a W&B artifact
pre_encoding = df_transform.to_csv(pre_encoding_path, index=False)
processed_data_artifact = wandb.Artifact("pre-encoding-file", type='dataset')
processed_data_artifact.add_file(pre_encoding_path)
wandb.log_artifact(processed_data_artifact)

# Correlation Analysis
corr_matrix = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
#plt.show()
corr_matrix_str = corr_matrix.to_string()
print(corr_matrix_str)
corr_matrix_plot_path = f"./{project}/artifacts/"
corr_matrix_csv = corr_matrix.to_csv(f"./{project}/artifacts/corr_matrix_{project}_correlation_matrix.csv", index=True)
wandb.log({"corr_matrix": corr_matrix})


# Step 3: Log the Correlation Matrix as an Artifact in W&B
art_path=f"./{project}/artifacts/corr_matrix_{project}_"
artifact = wandb.Artifact(name='correlation_matrix', type='dataset', description="Correlation matrix of the dataset")
artifact.add_file(f"./{project}/artifacts/corr_matrix_{project}_correlation_matrix.csv")
wandb.log_artifact(artifact)

# Optional: Visualize the Correlation Matrix
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix After Encoding')
plt.show()

print("Finishing WandB...")
if wandb.run is not None:
    final_csv = df_transform.to_csv(f"{project}/data/{datetime.now().strftime('%Y_%m_%d')}_{project}_v{version}.csv", index=True)
    final_data_artifact = wandb.Artifact("final_data", type='dataset')
    final_data_artifact.add_file(final_csv)
    wandb.log_artifact(final_data_artifact)
    wandb.finish()
    print("ALL DONE...")

