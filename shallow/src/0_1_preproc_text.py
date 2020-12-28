#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 18:29:06 2020
@author: grosati
"""
import pandas as pd

def clean_text_df(df, input_text_file='texto', stopwords=False):
    """
    Función para el procesamiento de texto
    1. pasa todo a minúscula
    2. elimina sitios web
    3. elimina puntuación
    4. elimins dígitos
    5. pasa todo a utf-8
    6. elimina espacios al inicip y al final
    7. elimina stopwords
    Devuelve archivo 0_data_proc.csv
    """
    text_raw = df[input_text_file]
    text_ed = text_raw.str.lower() # Pasa todo a minúsucula
    text_ed = text_ed.str.replace(r'^https?:\/\/.*[\r\n]*', r'_web_site_') # Elimina sitios web
    text_ed = text_ed.str.replace(r'[¡|¿|?|!|\'|"|#]',r'') # ELimina puntuación
    text_ed = text_ed.str.replace(r'[.|,|:|)|(|\|/]',r' ') # ELimina puntuación
    text_ed = text_ed.str.replace(r'[0-9]+',r'_digit_') # ELimina dígitos
    text_ed = text_ed.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    text_ed = text_ed.str.replace(r'\n', '').replace('\r', '') # Elimina espacios a izqueierda y derecha
    text_ed = text_ed.str.replace(r'  ', ' ').replace('   ', ' ').replace('    ', ' ')
    if stopwords == True:
        text_ed = text_ed.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    
    return(text_ed)

def proc_etapa0(df, type_proc='normal', stop_name=''):
    """
    Función que define cómo se codifica la capa 1.
    Devuelve archivo con columna etiqueta_etapa0 correspondiente
    """
    
    if type_proc == 'normal':
        df['etiqueta_etapa0'] = df['etiqueta_final'].replace({"POSITIVO" : 'R', 
                                                          "NEGATIVO" : 'R',
                                                          "NEUTRAL" :  'R',
                                                          "NECESIDADES OPERATIVAS": 'R',
                                                          "NO RELEVANTE" : 'NR'})
    
    if type_proc == 'pos_neg':
        df['etiqueta_etapa0'] = df['etiqueta_final'].replace({"POSITIVO" : 'R', 
                                                          "NEGATIVO" : 'R',
                                                          "NEUTRAL" :  'NR',
                                                          "NECESIDADES OPERATIVAS": 'NR',
                                                          "NO RELEVANTE" : 'NR'})
        
    if type_proc == 'necop_pos_neg':
        df['etiqueta_etapa0'] = df['etiqueta_final'].replace({"POSITIVO" : 'R', 
                                                          "NEGATIVO" : 'R',
                                                          "NEUTRAL" :  'NR',
                                                          "NECESIDADES OPERATIVAS": 'NEC_OP',
                                                          "NO RELEVANTE" : 'NR'})
    
    if type_proc == 'pos_neu_neg':
         df['etiqueta_etapa0'] = df['etiqueta_final'].replace({"POSITIVO" : 'R', 
                                                          "NEGATIVO" : 'R',
                                                          "NEUTRAL" :  'R',
                                                          "NECESIDADES OPERATIVAS": 'NR',
                                                          "NO RELEVANTE" : 'NR'})
    
    
    df.to_csv('./data/proc/data_' + type_proc + '_proc_' + stop_name +'.csv', index=False)

    return(df)
#%%
stop_words = open('./data/raw/stopwords.txt', 'r').read().split('\n')
data = pd.read_csv(r'./data/raw/raw_data.csv')

type_dataset='normal'

#%%
data = data[['id', 'texto', 'etiqueta1','etiqueta1a', 'etiqueta2', 'etiqueta2a',
'etiqueta3', 'etiqueta3a', 'etiqueta4', 'etiqueta4a', 'etiqueta5', 'etiqueta_final']]

#data['texto_ed'] = clean_text_df(data, 'texto')

data['etiqueta_final'] = data['etiqueta_final'].astype('str')
data['etiqueta_final'] = data['etiqueta_final'].str.upper()

data = data[['id', 'texto', 'etiqueta_final']]
#%% Agregamos data extra de negativos

data_extra = pd.read_csv(r'./data/raw/test_raw_data.csv')

data_extra = data_extra[['id', 'texto', 'etiqueta1','etiqueta2', 
'etiqueta3', 'etiqueta4', 'etiqueta_final']]

#['texto_ed'] = clean_text_df(data_extra, 'texto')
data_extra['etiqueta_final'] = data_extra['etiqueta_final'].astype('str')
data_extra['etiqueta_final'] = data_extra['etiqueta_final'].str.upper()
data_extra = data_extra[['id', 'texto', 'etiqueta_final']]

data_extra_neg = data_extra[data_extra['etiqueta_final'] == 'NEGATIVO']
#%%
data_final = pd.concat([data, data_extra_neg])    
#%%
data_final['texto_ed'] = clean_text_df(data_final, 'texto', stopwords=False)
t = proc_etapa0(data_final, type_proc='necop_pos_neg', stop_name='')
#%%
#%%

t.etiqueta_final.value_counts()
