# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import pandas as pd 
#import do pandas
dados = pd.read_csv('credit-data.csv')
#importa a base de dados
dados.describe()
#descreve dados estatisticos
dados.loc[dados['age']<0]
#localiza os registros com age menor que zero
dados.drop('age', 1, inplace = True)
#deleta a base de dados
dados.drop(dados[dados.age < 0].index, inplace=True)
#deleta linhas especificas
dados.mean()
#retorna média
dados.loc[dados.age < 0, 'age'] = 40.927700
#insere valor em campo especifico
pd.isnull(dados['age'])
#verifica se a coluna é nula
dados.loc[pd.isnull(dados['age'])]
#lista todas as colunas nulas
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