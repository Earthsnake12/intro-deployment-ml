from dvc import api
import pandas as pd
from io import StringIO
import sys
import logging

# esto es un registro para el manejo de info, errores, etc
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level= logging.INFO,
    datefmt= '%H:%M:%S',
    stream= sys.stderr
)

logger =  logging.getLogger(__name__)

logging.info("Fetching data..")

# abro los csv
movie_data_path = api.read("dataset/movies.csv", remote="tracker", encoding="utf8")
finantials_data_path = api.read("dataset/finantials.csv", remote="tracker", encoding="utf8")
opening_data_path = api.read("dataset/opening_gross.csv", remote="tracker", encoding="utf8")

# los leo y guardo en variables
movie_data = pd.read_csv(StringIO(movie_data_path))
fin_data = pd.read_csv(StringIO(finantials_data_path))
opening_data = pd.read_csv(StringIO(opening_data_path))

# aca estoy repitiendo lo del notebook
numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
movie_data = movie_data[numeric_columns + ['movie_title']]

fin_data = fin_data[['movie_title','production_budget','worldwide_gross']]

fin_movie_data = pd.merge(fin_data, movie_data, on= 'movie_title', how='left')
full_movie_data = pd.merge( opening_data,fin_movie_data, on = 'movie_title', how='left')

full_movie_data = full_movie_data.drop(['movie_title','gross'],axis=1)

full_movie_data.to_csv("dataset/full_data.csv", index=False)

logger.info("Data extraida y preparada")