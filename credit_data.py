# -*- coding: utf-8 -*-
"""
Created on Tue May 15 08:26:03 2019

@author: felipe
"""
import pandas as pd
import numpy as np
base = pd.read_csv('credit-data.csv')

base.loc[pd.isnull(base['age'])]

base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92
        
previsores = base.iloc[:, 1:4].values
meta = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, meta_treinamento, meta_teste = train_test_split(previsores, meta, test_size = 0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, meta_treinamento)

resultado = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(meta_teste, resultado)
previsao_template = confusion_matrix(meta_teste, resultado)
                 
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  