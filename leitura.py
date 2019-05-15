# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import pandas as pd 
#import do pandas
dados = pd.read_csv('credit-data.csv')
#importa a base de dados
dados.loc[dados.age < 0, 'age'] = 40.927700
#insere valor em campo especifico
previsores = dados.iloc[:, 1:4].values
classificadores = dados.iloc[:,4].values
#divide a base entre previsores e classificadores
from sklearn.impute import SimpleImputer
import numpy as np 
#import do SimpleImputer para realizar tratamento de dados faltantes
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])
#substitui os valores nulos pela média