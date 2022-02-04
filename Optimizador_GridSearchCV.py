…

# Librería necesaria para utilizar GridSearchCV
from sklearn.model_selection import GridSearchCV 

# Librería necesaria para algoritmo DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

…

# Se seleccionarán los rangos de los hiperparámetros a probar
parameters={'criterion': ['gini','entropy'], 'min_samples_split' : range(2,500,10),'max_depth': range(1, 120, 2),
                'min_samples_leaf': [1, 5, 10]}

clf_tree=DecisionTreeClassifier()

# buscamos la mejor opción
clf=GridSearchCV(clf_tree, parameters)

clf.fit(X, y)

ad = clf.best_estimator_

ad.fit(X, y)

…