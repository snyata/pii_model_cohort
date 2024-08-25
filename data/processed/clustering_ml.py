import os

import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import wandb

load_dotenv()



device = "mps" if torch.backends.mps.is_available() else "cpu"
# Initialize a W&B run
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="customer_personality_analysis", entity="orionai", name="clustering", job_type="train")

# Load the processed data
df_processed = pd.read_csv('df_processed.csv')

# Train-Test Split
X = df_processed.drop(columns=['target'])
y = df_processed['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Log dataset artifact to W&B
dataset_artifact = wandb.Artifact('processed_data', dataset_artifact='dataset')
dataset_artifact.add_file('df_processed.csv')
wandb.log_artifact(dataset_artifact)

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
kmeans_labels = kmeans.predict(X_test)
kmeans_silhouette = silhouette_score(X_test, kmeans_labels)

# Log KMeans results
wandb.log({"KMeans Silhouette Score": kmeans_silhouette})

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=2)
agg_labels = agg_clustering.fit_predict(X_test)
agg_silhouette = silhouette_score(X_test, agg_labels)

# Log Agglomerative Clustering results
wandb.log({"Agglomerative Clustering Silhouette Score": agg_silhouette})

# Summary of results
print("\nSummary of Clustering Results:")
print(f"KMeans Silhouette Score: {kmeans_silhouette:.2f}")
print(f"Agglomerative Clustering Silhouette Score: {agg_silhouette:.2f}")

# Log final model metrics and artifacts
wandb.log({
    "kmeans_silhouette_score": kmeans_silhouette,
    "agg_silhouette_score": agg_silhouette
})

# Optional: Save the trained models and log them as artifacts
k_model_artifact = wandb.Artifact('kmeans_model', type='model')
wandb.log_artifact(k_model_artifact)

agg_model_artifact = wandb.Artifact('agg_clustering_model', type='model')
wandb.log_artifact(agg_model_artifact)

# Finish the W&B run
wandb.finish()

# Optional: If you want to run hyperparameter sweeps with W&B
sweep_config = {
    'method': 'random',
    'parameters': {
        'n_clusters': {
            'values': [2, 3, 4, 5]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="customer-response-prediction")

def sweep_train():
    # Initialize a W&B run
    with wandb.init() as run:
        config = wandb.config
        
        # Run KMeans with the sweep's number of clusters
        kmeans_sweep = KMeans(n_clusters=config.n_clusters, random_state=42)
        kmeans_sweep.fit(X_train)
        kmeans_sweep_labels = kmeans.predict(X_test)
        kmeans_sweep_silhouette = silhouette_score(X_test, kmeans_sweep_labels)
        
        # Log the results
        wandb.log({"KMeans Silhouette Score": kmeans_sweep_silhouette})

# Run the sweep
wandb.agent(sweep_id, sweep_train)