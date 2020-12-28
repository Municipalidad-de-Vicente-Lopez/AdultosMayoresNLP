#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:31:05 2020

@author: grosati
"""
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
#%%
type_proc = 'necop_pos_neg'
type_stopw = ''
type_os = ''

#%%


file_path = r'./data/proc/data_' + type_proc + '_proc' + type_os + type_stopw + '.csv'


#%%
data = pd.read_csv(file_path,lineterminator='\n')
data.dropna(subset=['texto_ed'], inplace=True)
#%% test & train data unbalanced
train_data = data.sample(frac=.75, random_state=2247)
test_data = data.drop(train_data.index)

train_data.to_csv('./data/proc/train_' + type_proc + type_os + '_' + type_stopw + '.csv')
test_data.to_csv('./data/proc/test_' + type_proc + type_os + '_' + type_stopw + '.csv')

#%% index under & oversampled
rus = RandomUnderSampler(random_state=9797)
rus.fit_resample(train_data[['texto']], train_data['etiqueta_final'])

ros = RandomOverSampler(random_state=1244)
ros.fit_resample(train_data[['texto']], train_data['etiqueta_final'])

train_data_us = train_data.iloc[rus.sample_indices_,]
train_data_os = train_data.iloc[ros.sample_indices_,]
#%% save data

train_data_us.to_csv('./data/proc/train_data_us_nostop.csv')
train_data_os.to_csv('./data/proc/train_data_os_nostop.csv')

