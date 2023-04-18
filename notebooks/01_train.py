# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Movie Recommendations Using BERT Embeddings & Sentence Transformers
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC * In this notebook, we will create a simple movie recommendation engine using BERT Embeddings.
# MAGIC * We will use movie data from Wikipedia generate our embeddings.
# MAGIC * We'll then create embeddings for our query term and run semantic search using Sentence Transformers.

# COMMAND ----------

# DBTITLE 1,Prerequisites
# Install Sentence Transformers

!pip install -U sentence-transformers -q

# Download movie data

!python ../utils/data.py

# COMMAND ----------

# DBTITLE 1,Reading the data
import pandas as pd

df = pd.read_json("/tmp/movies.json")
df.head()

# COMMAND ----------

# DBTITLE 1,Data Cleaning and Generating Features
import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Converting to lower case and removing nulls

df_clean = df.dropna(subset = "extract")
df_clean.loc[:, "extract"] = df_clean.loc[:, "extract"].str.lower()
df_clean.loc[:, "title"] = df_clean.loc[:, "title"].str.lower()

# Combine items from cast and genres arrays into a string column

df_clean.loc[:, "actors_str"] = df_clean.loc[:, "cast"].apply(lambda x: " ".join([actor.lower() for actor in x]))
df_clean.loc[:, "genres_str"] = df_clean.loc[:, "genres"].apply(lambda x: " ".join([genre.lower() for genre in x]))

# Combining movie title, year, description and cast into one column.
# Notice that we use '[SEP]' between each field.
# This will be useful to properly create embeddings using BERT.

df_clean.loc[:, "full_desc"] = (
    df_clean.loc[:, "genres_str"]
    + "[SEP]"
    + df_clean.loc[:, "actors_str"]
)

# Sample

df_clean.sample(5)

# COMMAND ----------

from sentence_transformers import SentenceTransformer, util
import torch

# By using model.start_multi_process_pool() we can generate
# embeddings with multiple GPUs

model = SentenceTransformer('all-MiniLM-L6-v2')
pool = model.start_multi_process_pool()
sample_extracts = df_clean.reset_index(drop = True).full_desc.values
movie_embeddings = model.encode_multi_process(sample_extracts, pool = pool)

# COMMAND ----------

import numpy as np
from typing import List, Any
from matplotlib import pyplot as plt
from PIL import Image
import math
from io import BytesIO
import requests
from PIL import UnidentifiedImageError
import time

HEADERS = {
    'authority': 'upload.wikimedia.org',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
}

def find_top_match(embeddings: List[Any], search_term: str, top_k: int = 1) -> np.array:

    query_embedding = model.encode(search_term)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k = top_k)
    best_match_embedding = embeddings[top_results[1]]

    return best_match_embedding

def get_recommendations(embeddings: np.array, search_embedding: np.array, top_k: int = 5) -> pd.DataFrame:

    cos_scores = util.cos_sim(search_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k = top_k)

    idx = top_results[1].detach().numpy()
    idx = [item for item in idx if item != query_index]
    df_results = index_df.loc[idx, ["title", "extract", "thumbnail"]]
    df_results["score"] = top_results[0].detach().numpy()[:len(idx)]

    return df_results

def plot_thumbnails(
    search_term: str,
    recommendations: pd.DataFrame,
    nrows = 1,
    ncols = 5
):

    images = []

    recommendations = recommendations.sort_values("score", ascending = False)
    thumbnails = recommendations.thumbnail.values
    titles = recommendations.title.values
    scores = recommendations.score.values

    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20, 5))
    for i, thumbnail in enumerate(thumbnails):
        time.sleep(0.3)
        try:
            response = requests.get(
                url = thumbnail,
                headers = HEADERS
            )
            im = np.asarray(Image.open(BytesIO(response.content), formats = ["JPEG", "PNG"]))
            ax[i].imshow(im)
            ax[i].set_title(f"{titles[i]} ({scores[i]:.2f})")
        except UnidentifiedImageError:
            print(f"Error reading image: {thumbnail}")
    
    fig.suptitle(f"Recommendations for {search_term}")


search_term = "batman"
target_embedding = find_top_match(movie_embeddings, search_term = search_term)
recommendations = get_recommendations(movie_embeddings, search_embedding = target_embedding)
plot_thumbnails(search_term = search_term, recommendations = recommendations)

# COMMAND ----------

# DBTITLE 1,Register MLflow Model
import mlflow
import torch
import json
from sentence_transformers import SentenceTransformer

class MovieRecommendationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device = device)
        self.movies_embeddings = torch.load(context.artifacts["movies_embeddings"])
        self.movies_df = pd.read_json(context.artifacts["movies_df"])
    
    def _get_best_match(self, search_query: str):

        query_embedding = self.model.encode(search_query)
        cos_scores = util.cos_sim(query_embedding, self.movies_embeddings)[0]
        top_results = torch.topk(cos_scores, k = 1)
        best_match_index = top_results[1]
        best_match_embedding = self.movies_embeddings[best_match_index]

        return best_match_index, best_match_embedding

    def predict(self, context, model_input):

        search_query = model_input["search_query"]
        top_k = 10

        best_match_index, best_match_embedding = self._get_best_match(search_query)
        cos_scores = util.cos_sim(best_match_embedding, self.movies_embeddings)[0]
        top_results = torch.topk(cos_scores, k = top_k)

        idx = top_results[1].detach().numpy()
        idx = [item for item in idx if item != best_match_index]
        df_results = self.movies_df.loc[idx, ["title", "extract", "thumbnail", "genres", "year"]]
        df_results["score"] = top_results[0].detach().numpy()[:len(idx)]

        return json.loads(df_results.to_json(orient = "records"))

# COMMAND ----------

# DBTITLE 1,Quickly Testing the MLflow Model
def test_prediction():

    class Context(object):
        pass

    context = Context()
    context.artifacts = {
        "movies_embeddings": "/tmp/embeddings.pt",
        "movies_df": "/tmp/movies.json"
    }

    recommendation_model = MovieRecommendationModel()
    model_input = {"search_query": "batman"}

    recommendation_model.load_context(context = context)
    pred = recommendation_model.predict(context = context, model_input = model_input)
    return pred

test_prediction()

# COMMAND ----------

torch.save(torch.tensor(movie_embeddings), "/tmp/embeddings.pt")
df_clean.to_json("/tmp/movies.json", orient = "records")

with mlflow.start_run() as run:
  mlflow.pyfunc.log_model(
    artifacts = {
        "movies_embeddings": "/tmp/embeddings.pt",
        "movies_df": "/tmp/movies.json"
    },
    artifact_path = "recommendation_model",
    python_model = MovieRecommendationModel(),
    pip_requirements = [
        "sentence-transformers",
        "torch"
    ]
)

# COMMAND ----------


