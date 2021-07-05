# -*- coding: utf-8 -*-
"""
Functions para CAPM
Vamos a Guardar todas las Funciones 
"""
#Libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as lA


def load_time_series(ric, file_extension='csv'):
    #  Get Market Data
    path = 'C:\\Users\casa\\Downloads\\Finanzas Cuantitativas Py\\Bases de Datos\\' +ric+ '.' + file_extension
    if file_extension == 'csv':
        table_raw = pd.read_csv(path) 
    else:
        table_raw = pd.read_excel(path)
    #Create table of returns
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
    t['close'] = table_raw['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = table_raw['Close'].shift(1)
    t['returns_close'] = t['close']/t['close_previous'] -1 
    t = t.dropna()
    t = t.reset_index(drop=True)
    # Input for Jarque-Bera
    x = t['returns_close'].values           #Returns - Array
    x_str = 'Real_returns' + ric       #Etiquetas - Label RIC
    
    return x, x_str, t 

def plot_timeseries_price(t, ric):
    plt.figure()
    plt.plot(t['date'],t['close'])
    plt.title('Time Series Real Prices' + ric)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

def plot_histogram(x, x_str, plot_str, bins=100):
    plt.figure()
    plt.hist(x, bins)
    plt.title('Histogram ' + x_str)
    plt.xlabel(plot_str) 
    plt.show()
    
def synchronize_timeseries(ric, benchmark, file_extension='csv'):
    # Loading data from csv or Excel file 
    x1, str1, t1 = load_time_series(ric, file_extension)
    x2, str2, t2 = load_time_series(benchmark, file_extension)
    #Synchronize Timetamps
    timetamp1 = list(t1['date'].values)
    timetamp2 = list(t2['date'].values)
    timetamps =list(set(timetamp1) & set(timetamp2))
    # Synchronised Time Series for x1 or ric
    t1_sync = t1[t1['date'].isin(timetamps)]
    t1_sync.sort_values(by='date', ascending=True)
    t1_sync = t1_sync.reset_index(drop=True)
    #Synchronised Time Series for x2 or Benchmark
    t2_sync = t2[t2['date'].isin(timetamps)]
    t2_sync.sort_values(by='date', ascending=True)
    t2_sync = t2_sync.reset_index(drop=True)
    #Table of Returns for ric and benchmark
    t = pd.DataFrame()
    t['data'] = t1_sync['date']
    t['price_1'] = t1_sync['close']
    t['price_2'] = t2_sync['close']
    t['return_1'] = t1_sync['returns_close']
    t['return_2'] = t2_sync['returns_close']
    #Compute Vectors of Returns
    y = t['return_1'].values
    x = t['return_2'].values 
    
    return x, y, t
    