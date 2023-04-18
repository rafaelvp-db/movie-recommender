# Databricks notebook source
!pip install sentence-transformers -q

# COMMAND ----------

# DBTITLE 1,Testing Predictions Locally
import mlflow

run_id = "fb352c4f6e0b4a1bad085257bd35d90c"
logged_model = f'runs:/{run_id}/recommendation_model'
recommendation_model = mlflow.pyfunc.load_model(logged_model)

recommendation_model.predict({"search_query": "batman"})

# COMMAND ----------

# DBTITLE 1,Deploy to Model Registry
model_name = "recommendation_model"

model_info = mlflow.register_model(
    model_uri = f"runs:/{run_id}/recommendation_model",
    name = model_name
)

model_version = model_info.version

# COMMAND ----------

# DBTITLE 1,Promote to Production
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name = model_name,
    version = model_version,
    stage = "Production",
    archive_existing_versions = True
)

# COMMAND ----------


