#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:58:40 2020

@author: grosati
"""
#%%
type_proc = 'necop_pos_neg'
type_stopw = ''
type_os = ''
#%%
import pandas as pd
import numpy as np

def gen_fasttext_data(data, 
                      label='etiqueta_final', 
                      features='texto_ed',
                      type_proc=type_proc):
    
    if label == 'etiqueta_final':
        data = data[data['etiqueta_etapa0'] == 'R']
        
    data_ft = '__label__' + data[label] + ' ' + data[features]
    return(data_ft)
    
    
#%%

file_path_train = './fasttext/data/data_' + type_proc + type_os + type_stopw +'_proc.csv'
#file_path_test = './fasttext/data/proc/test_' + type_proc + type_os + type_stopw +'.csv'
#cv_results_path = './model_evaluation/l1_grid_' + type_proc + type_os + '_nostop.csv'
#model_out_path = './models/l1_' + type_proc + type_os + '_'
#model_eval_path = './model_evaluation/l1_metrics_' + type_proc + type_os + '_nostop.csv'

data_out_path = './fasttext/data/train_' + type_proc + type_os + type_stopw + '.txt'
#%%
data = pd.read_csv(file_path_train, lineterminator='\n')
data.dropna(subset=['texto_ed'], inplace=True)

#%%
data.etiqueta_final.replace('NECESIDADES OPERATIVAS', 'NECOP', inplace=True)
data.etiqueta_etapa0.replace('NEC_OP', 'NECOP', inplace=True)
data.etiqueta_final.replace('NO RELEVANTE', 'NR', inplace=True)


#%% generamos test data
train_data = data.sample(frac=.80, random_state=79778)
test_data = data.drop(train_data.index)
#%%
test_data.to_csv('./fasttext/data/data_final/test_' + type_proc + type_os + type_stopw + '.csv')
#%% generamos validation data
train_data_ = train_data.sample(frac=.85, random_state=2358)
validation_data = train_data.drop(train_data_.index)
#%%
train_data_ft = gen_fasttext_data(train_data_, 'etiqueta_etapa0', 'texto_ed')
validation_data_ft = gen_fasttext_data(validation_data, 'etiqueta_etapa0', 'texto_ed')
test_data_ft = gen_fasttext_data(test_data, 'etiqueta_etapa0', 'texto_ed')

#%%
np.savetxt('./fasttext/data/data_final/0_train_' + type_proc + type_os + type_stopw + '.txt', 
           train_data_ft.values, fmt = "%s")

np.savetxt('./fasttext/data/data_final/0_validation_' + type_proc + type_os + type_stopw + '.txt', 
           validation_data_ft.values, fmt = "%s")

np.savetxt('./fasttext/data/data_final/0_test_' + type_proc + type_os + type_stopw + '.txt', 
           test_data_ft.values, fmt = "%s")

#%%
train_data_ft = gen_fasttext_data(train_data_, 'etiqueta_final', 'texto_ed')
validation_data_ft = gen_fasttext_data(validation_data, 'etiqueta_final', 'texto_ed')
test_data_ft = gen_fasttext_data(test_data, 'etiqueta_final', 'texto_ed')

#%%
np.savetxt('./fasttext/data/data_final/1_train_' + type_proc + type_os + type_stopw + '.txt', 
           train_data_ft.values, fmt = "%s")

np.savetxt('./fasttext/data/data_final/1_validation_' + type_proc + type_os + type_stopw + '.txt', 
           validation_data_ft.values, fmt = "%s")

np.savetxt('./fasttext/data/data_final/1_test_' + type_proc + type_os + type_stopw + '.txt', 
           test_data_ft.values, fmt = "%s")

