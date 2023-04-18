# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Set up notebook parameters
# MAGIC 
# MAGIC #### Parameters:
# MAGIC 
# MAGIC - `endpoint_name`: The name of the endpoint you wish to create with Inference Tables enabled.
# MAGIC - `dbfs_table_path`: The DBFS root path at which Inference Tables for this endpoint will be written. Each Inference Table will be created at the path `<dbfs_table_path>/<endpoint_name>`.
# MAGIC - `served_model_name`: The name of a served model to serve from this endpoint. You may add other served models later.
# MAGIC - `served_model_version`: The version of the above served model to serve from this endpoint.
# MAGIC - `served_model_workload_size`: The workload size of the served model.
# MAGIC - `served_model_scale_to_zero`: Whether the served model should scale to zero.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("endpoint_name", "")
dbutils.widgets.text("dbfs_table_path", "")
dbutils.widgets.text("served_model_name", "")
dbutils.widgets.text("served_model_version", "")
dbutils.widgets.dropdown("served_model_workload_size", "Small", ["Small", "Medium", "Large"])
dbutils.widgets.dropdown("served_model_scale_to_zero", "False", ["True", "False"])

# COMMAND ----------

endpoint_name = dbutils.widgets.get("endpoint_name")
dbfs_table_path = dbutils.widgets.get("dbfs_table_path")
served_model_name = dbutils.widgets.get("served_model_name")
served_model_version = dbutils.widgets.get("served_model_version")
served_model_workload_size = dbutils.widgets.get("served_model_workload_size")
served_model_scale_to_zero = bool(dbutils.widgets.get("served_model_scale_to_zero"))

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print("endpoint_name:", endpoint_name)
print("dbfs_table_path:", dbfs_table_path)
print("served_model_name:", served_model_name)
print("served_model_version:", served_model_version)
print("served_model_workload_size:", served_model_workload_size)
print("served_model_scale_to_zero:", served_model_scale_to_zero)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create serving endpoint with Inference Logging
# MAGIC 
# MAGIC This creates a serving endpoint with Inference Tables enabled.
# MAGIC 
# MAGIC **Note:** You can only enable Inference Tables at endpoint creation time.

# COMMAND ----------

import json
import requests

data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": served_model_name,
                "model_version": served_model_version,
                "workload_size": served_model_workload_size,
                "scale_to_zero_enabled": served_model_scale_to_zero,
            }
        ]
    },
    "inference_table_config": {
        "dbfs_destination_path": dbfs_table_path
    }
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/preview/serving-endpoints",
    json=data,
    headers=headers
)
print("Response status:", response.status_code)
print("Reponse text:", response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Get endpoint status
# MAGIC 
# MAGIC Get the status of your endpoint. Verify that Inference Table logging is enabled.

# COMMAND ----------

data = {
    "name": endpoint_name
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

response = requests.get(
    url = f"{API_ROOT}/api/2.0/preview/serving-endpoints/{endpoint_name}",
    json = data,
    headers = headers
)

print(response.status_code, response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test your endpoint
# MAGIC 
# MAGIC Try querying your endpoint!
# MAGIC 
# MAGIC **Note:** Logs will arrive in your Inference Table 5-10 minutes after the endpoint invocation.

# COMMAND ----------

import random
import uuid
import json

data = {
    "instances": [{"search_query": "stanley kubrick"}],
    "inference_id": str(uuid.uuid4())
}

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations",
    json=data,
    headers=headers
)

print("Reponse text:", json.dumps(response.json()["predictions"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test the raw Inference Log Delta table
# MAGIC 
# MAGIC Query your raw Inference Log Delta table!
# MAGIC 
# MAGIC **Note:** A log will arrive in your Inference Table about 5-10 minutes after the model invocation.

# COMMAND ----------

from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.getOrCreate()
df = spark.sql(f"select * from delta.`{dbfs_table_path}/{endpoint_name}` limit 1000")
display(df)

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    ArrayType,
    StringType,
    IntegerType,
    DoubleType
)

def expand_column(df, parent_column, child_field, schema):

    expanded_df = df.withColumn(f"{parent_column}_", F.from_json(F.col(parent_column), schema = schema))
    expanded_df = expanded_df \
        .select(f"{parent_column}_.{child_field}", "*") \
        .withColumn(child_field, F.col(f"{parent_column}_.{child_field}").getItem(0))
    expanded_df = expanded_df.select(f"{child_field}.*", "*")

    return expanded_df


def expand_df(df):

    element_request = StructType([
        StructField('search_query', StringType(), True)
    ])

    element_response = StructType([
        StructField('title', StringType(), True),
        StructField('extract', StringType(), True),
        StructField('thumbnail', StringType(), True),
        StructField('year', IntegerType(), True),
        StructField('score', DoubleType(), True),
    ])

    schema_request = StructType([
        StructField('instances', ArrayType(elementType = element_request), nullable = False)
    ])

    schema_response = StructType([
        StructField('predictions', ArrayType(elementType = element_response), nullable = False)
    ])

    expanded_df = expand_column(df, "request", "instances", schema = schema_request)
    expanded_df = expand_column(expanded_df, "response", "predictions", schema = schema_response)

    return expanded_df

# COMMAND ----------

# DBTITLE 1,Visualizing Most Recommended Years
expanded_df = expand_df(df)
display(expanded_df)

# COMMAND ----------

# DBTITLE 1,Visualizing Most Popular Queries
display(expanded_df)
