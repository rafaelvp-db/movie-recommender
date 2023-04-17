# Databricks notebook source
!pip install -U sentence-transformers -q

# COMMAND ----------

!python ../utils/data.py

# COMMAND ----------

import pandas as pd

df = pd.read_json("/tmp/movies.json")
df.head()

# COMMAND ----------

df_clean = df.dropna(subset = "extract")
df_clean["extract"] = df_clean["extract"].str.lower()
df_clean["title"] = df_clean["title"].str.lower()
df_clean["actors_str"] = df_clean["cast"].apply(lambda x: " ".join([actor.lower() for actor in x]))
df_clean["genres_str"] = df_clean["genres"].apply(lambda x: " ".join([genre.lower() for genre in x]))
df_clean["full_desc"] = df_clean["title"] + "[SEP]" + df_clean["year"].astype(str) + "[SEP]" + df_clean["genres_str"] + "[SEP]" + df_clean["actors_str"] #+ "[SEP]" + df_clean["extract"] #+ 
df_clean.sample(5).full_desc

# COMMAND ----------

batman_df = df_clean[(df_clean.extract.str.contains("batman")) & (df_clean.year > 1980)]
batman_df.head()

# COMMAND ----------

from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('all-MiniLM-L6-v2')

sample_extracts = df_clean.reset_index(drop = True).full_desc.values
movie_embeddings = embedder.encode(sample_extracts, convert_to_tensor = True)

# COMMAND ----------

batman_extract = batman_df[(batman_df.year == 2022)].full_desc.values[0]
query_embedding = embedder.encode(batman_extract, convert_to_tensor = True)

top_k = 10

cos_scores = util.cos_sim(query_embedding, movie_embeddings)[0]
top_results = torch.topk(cos_scores, k = top_k)

for score, idx in zip(top_results[0], top_results[1]):
  print(f"### Movie: {sample_extracts[idx].split('[SEP]')[:3]} ---> Score: {score}")

# COMMAND ----------


