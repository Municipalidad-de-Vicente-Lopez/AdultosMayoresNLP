#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:50:03 2020

@author: grosati
"""

## Funciones auxiliares

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
    Devuelve archivo string editado
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


def predict_partial(X, layer):
    y_preds = []
    y_preds_proba = []
    for index, value in X.items():
        preds_ = layer.predict(value)
        preds = preds_[0][0]
        pred_proba = preds_[1][0]
        y_preds.append(preds)
        y_preds_proba.append(pred_proba)
    return(np.array(y_preds), np.array(y_preds_proba))


def predict_full(layer_0, layer_1, data, text_col='texto_ed'):
    X0 = data[text_col]
    
    y_preds_l0, y_preds_proba_l0 = predict_partial(X0, layer_0)
    #y_preds_final = np.array(["__label__NR" for i in range(len(y_preds_l0))]).astype('object')
    y_preds_final = np.copy(y_preds_l0).astype('object')
    
    y_preds_proba_final = y_preds_proba_l0
    
    filt = np.array([True if '__label__R' in i else False for i in y_preds_l0])
    
    X1 = data.loc[filt, text_col]
    y_preds_final[filt], y_preds_proba_final[filt] = predict_partial(X1, layer_1)
    return(y_preds_final,y_preds_proba_final)