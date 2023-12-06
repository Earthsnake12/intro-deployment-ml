from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import logging
import sys
import numpy as  np
import pandas as pd

from utils import update_model, save_simple_metrics_report, get_model_performance_test_set

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level= logging.INFO,
    datefmt= '%H:%M:%S',
    stream= sys.stderr
)

logger =  logging.getLogger(__name__)

logging.info("Cargando data...")
data = pd.read_csv("dataset/full_data.csv")

# repite el notebook
logging.info("Cargando modelo...")
model = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')), #si encuentra valores #n/a los reemplaza por la media
    ('core_model', GradientBoostingRegressor())
])

logging.info("Separando data en train y test...")
X = data.drop(['worldwide_gross'], axis = 1)
y = data['worldwide_gross']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.35,random_state= 42)

logging.info("Preparando los hyperparametros...")
param_tunning = {'core_model__n_estimators': range(20,501,20)} 

grid_search= GridSearchCV(model,
                          param_grid = param_tunning,
                          scoring='r2',
                          cv=5) 

logging.info("Buscando los mejores parametros...")
grid_search.fit(X_train, y_train)

logging.info("Validacion cruzada del mejor modelo...")
final_result = cross_validate(grid_search.best_estimator_,X_train,y_train,return_train_score=True,cv=7)

train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])
assert train_score > 0.7
assert test_score > 0.65
logging.info(f'Train Score: {train_score}')
logging.info(f'Test Score: {test_score}')

logging.info("Actualizando modelo...")
update_model(grid_search.best_estimator_)

logging.info("Generando reporte de modelo...")
validation_score = grid_search.best_estimator_.score(X_test, y_test)
save_simple_metrics_report(train_score, test_score, validation_score, grid_search.best_estimator_)

logging.info("Generando grafica de modelo...")
Y_test_pred = grid_search.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, Y_test_pred)

logging.info("Entrenamiento completado")

