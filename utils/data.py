"""Helper script to download Wikipedia movie data."""

import requests
import time
import pandas as pd
import logging

logging.basicConfig(level = "DEBUG")
logger = logging.getLogger(__name__)

BASE_URL = "https://github.com/prust/wikipedia-movie-data/raw/master/movies-{}s.json"

def get_movie_data(start: int = 1960, end: int = 2020, output_path: str = "/tmp"):

    logger.info(f"Downloading movie info from {start} to {end}")
    decades = [start + (i * 10) for i in range(int((end-start)/10)+1)]
    list_df = []

    for decade in decades:
        response = requests.get(BASE_URL.format(decade))
        json_movies = response.json()
        df = pd.DataFrame.from_dict(json_movies)
        logger.info(f"DF: {df.head()}")
        list_df.append(df)

    output_df = pd.concat(list_df, axis = 0)
    json_path = f"{output_path}/movies.json"
    output_df.to_json(json_path, orient = "records")
    logger.info(f"Wrote movie data to {json_path}")
    

if __name__ == "__main__":
     get_movie_data()

