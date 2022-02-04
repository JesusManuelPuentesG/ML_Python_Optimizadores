…

# Librerías necesarias para utilizar Hyperopt
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin

# Librería necesaria para algoritmo Random Forest
from sklearn.ensemble import RandomForestClassifier

# Parámetros
MAX_EVALS = 50
N_FOLDS = 10

# Función objetivo para optimización de Random Forest
def objectiveRF(params, n_folds = N_FOLDS):
    """Función Objetivo para la optimización de hiperparámetros en Random Forest"""

    # Se implementa la validación cruzada de n_capas con
    # hiperparámetros
    clf = RandomForestClassifier(**params,n_estimators=20)
    scores = cross_val_score(clf, X_test, y_test, cv=5, scoring='f1_macro')

    # Se extrae el mejor resultado
    best_score = max(scores)

    # La función Loss deber ser minimizada
    loss = 1 - best_score

    # Devolvemos toda la información relevante
    return {'loss': loss, 'params': params, 'status': STATUS_OK}
    	
…

rf = RandomForestClassifier(n_estimators=20)

# especificamos parámetros a muestrear
space = {
    	'max_depth_p': hp.choice('max_depth', [3, None]),
    	'max_features_p' : hp.choice('max_features', range(1, 11)),
    	'min_samples_split_p' : hp.choice('min_samples_split', range(2, 11)),
    	'bootstrap_p' : hp.choice('bootstrap', [True, False]),
    	'criterion_p' : hp.choice('criterion', ['gini', 'entropy'])
        }

tpe_algorithm = tpe.suggest
bayes_trials = Trials()
best = fmin(fn = objectiveRF, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)

rf = RandomForestClassifier(
			max_depth = best['max_depth_p'],
			max_features = best['max_features_p'],
			min_samples_split = best['min_samples_split_p'],
			bootstrap = best['bootstrap_p'],
			criterion = best['criterion_p'])



rf.fit(X, y)

…
