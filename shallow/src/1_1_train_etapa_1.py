#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:32:05 2020

@author: grosati
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 19:16:22 2020

@author: grosati
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.svm import SVC  
import os

import joblib

# Clase que switchea entre diferentes estimadores

class ClfSwitcher(BaseEstimator):
    """
    Clase que switchea entre diferentes estimadores
    """
    def __init__(self, estimator = SGDClassifier(),):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
    
    
def get_paths(d):
    list_paths = []
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        if os.path.isfile(full_path):
            list_paths.append(full_path)
    return(list_paths)

def get_best_params(gscv, score_column='mean_test_recall'):
    """
    Función que extrae los mejores parámetros de cada tipo de modelo
    """
    cv_res = pd.DataFrame(gscv.cv_results_)
    cv_res['model'] = cv_res['param_clf__estimator'].astype('str').str.extract('([^\(:]+)')
    best_models_index = cv_res.reset_index().groupby(['model'])[score_column].idxmax()
    best_models = cv_res.iloc[best_models_index]
    return(best_models.params.to_list())

#%%

type_proc = 'pos_neg'
type_os = ''

file_path = './data/proc/train_' + type_proc + type_os +'_nostop.csv'
cv_results_path = './model_evaluation/l1_grid_' + type_proc + type_os + '_nostop.csv'
model_out_path = './models/l1_' + type_proc + type_os + '_'
model_eval_path = './model_evaluation/l1_metrics_' + type_proc + type_os + '_nostop.csv'


#%% Importa datos
train_data = pd.read_csv(file_path, lineterminator='\n')
train_data.dropna(subset=['texto_ed'], inplace=True)

#%% Lista de parámetros de cada modelo (con el preprocesamiento incluido)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', ClfSwitcher()),
])

parameters = [
    {
        'tfidf__max_df': (0.9, 1.0),
        'tfidf__min_df': (0.0, 0.05, 0.1),
        'tfidf__norm': ('l1','l2', None),
        'tfidf__use_idf': (True, False),
        'tfidf__stop_words': [None],

        'clf__estimator': [SGDClassifier()], # SVM if hinge loss / logreg if log loss
        'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
        'clf__estimator__alpha': (0.0001, 0.001, 0.01, 0.1, 1.0, 10),
        'clf__estimator__max_iter': [100],
        'clf__estimator__tol': [1e-5],
        'clf__estimator__loss': ['log'],
        },
    {
        'tfidf__max_df': (0.9, 1.0),
        'tfidf__min_df': (0.0, 0.05, 0.1),        
        'tfidf__norm': ('l1','l2', None),
        'tfidf__use_idf': (True, False),
        'tfidf__stop_words': [None],       
        'clf__estimator': [MultinomialNB()],
        'clf__estimator__alpha': (1e-2, 1e-3, 1e-1),
    },
    {
        'tfidf__max_df': (0.9, 1.0),
        'tfidf__min_df': (0.0, 0.05, 0.1),
        'tfidf__norm': ('l1','l2', None),
        'tfidf__use_idf': (True, False),
        'tfidf__stop_words': [None],
        'clf__estimator': [RandomForestClassifier()],
        'clf__estimator__n_estimators': [500],
        'clf__estimator__max_depth': (1, 5, 10, 20, 30, 50),
        'clf__estimator__max_features': (0.10, 0.25, 0.5, 0.75, 1.0),
     },
    
        {
        'tfidf__max_df': (0.9, 1.0),
        'tfidf__min_df': (0.0, 0.05, 0.1),
        'tfidf__norm': ('l1','l2', None),
        'tfidf__use_idf': (True, False),
        'tfidf__stop_words': [None],
        'clf__estimator': [SVC()],
        'clf__estimator__max_iter':[-1],
        'clf__estimator__C': (0.1, 1, 10, 100),
        'clf__estimator__gamma': (1,0.1,0.01,0.001),
        'clf__estimator__kernel': ['rbf', 'poly', 'sigmoid']
    },
]
#%%
data_ = train_data[train_data['etiqueta_etapa0'] != 'NR']

X = data_['texto_ed']
y = data_['etiqueta_final']

    
#%% Train modelos etapa 1
scoring = {'recall':'recall_macro',
           'prec': 'precision_macro'}

gscv = GridSearchCV(pipeline, parameters, 
                    cv=StratifiedKFold(n_splits=2, shuffle=True), 
                    n_jobs=-1, 
                    return_train_score=True,
                    scoring=scoring,
                    refit=False,
                    verbose=3)

gscv.fit(X, y)
results_ = pd.DataFrame(gscv.cv_results_)
results_.to_csv(cv_results_path)
#%%
best_params = get_best_params(gscv)
#%%
n_splits = 5
metrics = ['f1_micro', 'f1_macro', 'f1_weighted',
            'precision_micro', 'precision_macro', 'precision_weighted',
            'recall_micro', 'recall_macro', 'recall_weighted']

dump_file = True
scores_total = []

for i in range(0,len(best_params)):
   
    pipeline_best = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', ClfSwitcher()),
    ])
    
    name = best_params[i]['clf__estimator'].__class__.__name__
    pipeline_best.set_params(**best_params[i])
    print('\n')
    print('---------------------------------')
    print('Fitting ' + name + '...')
   
    fit = pipeline_best.fit(X, y)
    
    scores = cross_validate(pipeline_best, 
                            X, 
                            y, 
                            cv=StratifiedKFold(n_splits=n_splits),
                            scoring=metrics,
                            return_train_score=False)
    
    cv_score = {k : np.mean(scores[k]) for k in scores}
    print('\n')
    print('Score:' + '\n')
    print(pd.Series(cv_score))    
    scores_total.append({'model' : name, 'metrics' : cv_score})
 
    #ACA SERIALIZAR EL FIT
    if dump_file == True:
        joblib.dump(fit, model_out_path + name + '.joblib')
        print('\n')
        print(name + ' serialized')
        print('----------------------------------')
#%%
scores_total = pd.DataFrame(scores_total)
scores_total = pd.concat([scores_total.drop(['metrics'], axis=1),
           scores_total['metrics'].apply(pd.Series)], axis=1).set_index('model').T
scores_total.to_csv(model_eval_path)


