# -*- coding: utf-8 -*-
"""
Create classes

"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress

# Import our own Function and Class files and Reload
import stream_functions
importlib.reload(stream_functions)

class jarque_bera_test():
    
    def __init__(self, x, x_str):
        self.returns = x
        self.str_name = x_str 
        self.size = len(x) #Size of returns
        self.round_digits = 4
        self.mean = 0.0
        self.stdev = 0.0
        self.skew = 0.0
        self.kurt = 0.0
        self.median = 0.0
        self.var_95 = 0.0
        self.cvar_95 = 0.0
        self.jarque_bera = 0.0
        self.p_value = 0.0
        self.is_normal = 0.0
        
    def compute(self):
        self.mean = np.mean(self.returns)
        self.stdev = np.std(self.returns)    #Volatility 
        self.skew = skew(self.returns)
        self.kurt = kurtosis(self.returns) # excess kurtosis 
        self.sharpe = self. mean / self.stdev * np.sqrt(252)
        self.median = np.median(self.returns)
        self.var_95 = np.percentile(self.returns,5)
        self.cvar_95 = np.mean(self.returns[self.returns <= self.var_95]) 
        self.jarque_bera = self.size/6*(self.skew**2 + 1/4*self.kurt**2)
        self.p_value = 1 - chi2.cdf(self.jarque_bera, df=2) #Degree Freedom 
        self.is_normal = (self.p_value > 0.05 ) #Equivalenty x_jarque_bera < 6
        
        
    def __str__(self):
        str_self = self.str_name + ' / size ' + str(self.size) + '\n' + self.plot_str()
        return str_self
    
        
    def plot_str(self):
        # Print Metrics in Graph
        round_digits = 4
        plot_str = 'mean ' + str(np.round(self.mean,round_digits))\
            + ' / std dev ' + str(np.round(self.stdev,round_digits))\
            + ' / skewness ' + str(np.round(self.skew,round_digits))\
            + ' / kurtosis ' + str(np.round(self.kurt,round_digits))\
            + ' / Sharpe ratio ' + str(np.round(self.sharpe,round_digits)) + '\n'\
            + 'VaR 95% ' + str(np.round(self.var_95,round_digits))\
            + ' / CVaR 95% ' + str(np.round(self.cvar_95,round_digits))\
            + ' / Jarque_Bera ' + str(np.round(self.jarque_bera,round_digits))\
            + ' / p_value ' + str(np.round(self.p_value,round_digits))\
            + ' / is_normal ' + str(self.is_normal)
        return plot_str


class capm_manager():
    
    def __init__(self, ric, benchmark):
        self.nb_decimals = 4
        self.ric = ric
        self.benchmark = benchmark
        self.x = []
        self.y = []
        self.t = []
        self.beta = 0.0
        self.alpha = 0.0
        self.p_value = 0.0
        self.null_hypothesis = False
        self.r_value = 0.0
        self.r_squared = 0.0
        self.predictor_linreg = []
        
    def __str__(self):
        str_self = 'linear regression | ric ' + self.ric\
            + '| benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept)' + str(self.alpha)\
            + '| beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + '| null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'r-value ' + str(self.r_value)\
            + ' | r_squared ' + str(self.r_squared)
        
        return str_self
        
    def load_timeseries(self):
        #Load timeseries and synchronize 
        self.x, self.y, self.t = stream_functions.synchronize_timeseries(self.ric, self.benchmark)
        
    def compute(self):
        # Linear Regression of ric  with respect to benchmark
        slope, intercept, r_value, p_value, std_err = linregress(self.x,self.y)
        self.beta = np.round(slope, self.nb_decimals)
        self.alpha = np.round(intercept, self.nb_decimals)
        self.p_value = np.round(p_value, self.nb_decimals)
        self.null_hypothesis = p_value > 0.05 #p_value<0.05 Reject null hypothesis
        self.r_value = np.round(r_value, self.nb_decimals) #Correlation Coefficient
        self.r_squared = np.round(r_value**2, self.nb_decimals) # Pct of Variance of "y" explained by "x"
        self.predictor_linreg = self.alpha + self.beta*self.x
        
    def scatterplot(self):
        #Scatterplot of returns
        str_title = 'Scaterplot of returns' + '\n' + self.__str__()

        plt.figure()
        plt.title(str_title)
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.predictor_linreg, color='green')
        plt.ylabel(self.ric)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()
        
    def plot_normalized(self):
        price_ric = self.t['price_1']
        price_benchmark = self.t['price_2']
        plt.figure(figsize=(12,5))
        plt.title('Time series of price | Normalized at 100')
        plt.xlabel('Time')
        plt.ylabel('Normalized Prices')
        price_ric = 100 * price_ric / price_ric[0]
        price_benchmark = 100 * price_benchmark / price_benchmark[0]
        plt.plot(price_ric, color='blue', label=self.ric)
        plt.plot(price_benchmark, color='red', label=self.benchmark)
        plt.legend(loc=0)
        plt.grid()
        plt.show()
        
    def plot_dual_axes(self):
        plt.figure(figsize=(12,5))
        plt.title('Time Series of Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        ax1 = self.t['price_1'].plot(color='blue', grid=True, label= self.ric)
        ax2 = self.t['price_2'].plot(color='red', grid=True, secondary_y=True, label= self.benchmark)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()