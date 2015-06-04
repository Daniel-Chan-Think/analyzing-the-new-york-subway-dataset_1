# -*- coding: utf-8 -*-
"""
@author: Daniel
"""

import numpy as np 
import pandas as pd
import statsmodels.api as sm

#from ggplot import *
import matplotlib.pyplot as plt

#import scipy


file1 = 'C:\Users\Daniel\Downloads\improved-dataset\\turnstile_weather_v2.csv'
file2 = 'C:\Users\Daniel\Downloads\\turnstile_data_master_with_weather.csv'
turnstile_weather_v2 = pd.read_csv(file1)

# scipy.stats.mannwhitneyu returns p-value as nan, but returns it normally when several rows of data set are discarded
turnstile_weather_v2 = turnstile_weather_v2[6:len(turnstile_weather_v2['rain'])]

### Data Wangling


### Data Visualization


### Data Analyzing

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    
    This can be the same code as in the lesson #3 exercise.
    """
    
    #features = sm.add_constant(features) 
    n = len(features[:,0])
    ones = np.ones((n,1))
    features = np.hstack((ones, features))
    
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:len(results.params)]
       
    return intercept, params

def predictions(features_array, params, intercept):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe. 
    We recommend that you don't use the EXITSn_hourly feature as an input to the 
    linear model because we cannot use it as a predictor: we cannot use exits 
    counts as a way to predict entry counts. 
    
    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in 
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with 
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.

    '''
    
    predictions = intercept + np.dot(features_array, params)
    return predictions
    
def compute_r_squared(data, predictions):
    r_squared = 1 - (np.square(data - predictions)).sum() / (np.square(data - data.mean())).sum()
    return r_squared


# Select Features (try different features!)
'''
features = turnstile_weather_v2[['rain', 'fog', \
        'meanprecipi', 'pressurei', 'meantempi', 'wspdi', \
        'day_week', 'weekday', 'hour']]
'''
features = turnstile_weather_v2[['rain']]
'''
chosen features: ['rain' 'fog' 'precipi' 'pressurei' 'tempi' 'wspdi']
num of chosen features: 6
R^2: 0.548063831692

chosen features: ['rain' 'fog']
num of chosen features: 2
R^2: 0.545827778104

chosen features: ['rain' 'fog']
num of chosen features: 2
R^2: 0.547518525878
'''

print "chosen features:", features.columns.values
features_num = len(features.columns.values)
print "num of chosen features:", features_num

dummy_units = pd.get_dummies(turnstile_weather_v2['fog'], prefix='fog')
features = features.join(dummy_units)
dummy_units = pd.get_dummies(turnstile_weather_v2['precipi'], prefix='precipi')
features = features.join(dummy_units)
dummy_units = pd.get_dummies(turnstile_weather_v2['pressurei'], prefix='pressurei')
features = features.join(dummy_units)
dummy_units = pd.get_dummies(turnstile_weather_v2['tempi'], prefix='tempi')
features = features.join(dummy_units)
dummy_units = pd.get_dummies(turnstile_weather_v2['wspdi'], prefix='wspdi')
features = features.join(dummy_units)

# Add 'UNIT' to features using dummy variables.
dummy_units = pd.get_dummies(turnstile_weather_v2['UNIT'], prefix='unit')
features = features.join(dummy_units)
# Add 'conds' to features using dummy variables.
dummy_units = pd.get_dummies(turnstile_weather_v2['conds'], prefix='conds')
features = features.join(dummy_units)

dummy_units = pd.get_dummies(turnstile_weather_v2['weekday'], prefix='weekday')
features = features.join(dummy_units)
dummy_units = pd.get_dummies(turnstile_weather_v2['day_week'], prefix='day_week')
features = features.join(dummy_units)
dummy_units = pd.get_dummies(turnstile_weather_v2['hour'], prefix='hour')
features = features.join(dummy_units)

features_array = features.values
values = turnstile_weather_v2['ENTRIESn_hourly']
values_array = values.values

##### Perform linear regression
intercept, params = linear_regression(features_array, values_array)
    
##### do prediction
predictions = predictions(features, params, intercept)
print "R^2:", compute_r_squared(values, predictions)

print "intercept:", intercept
print "params for the chosen features:", params[0:features_num]

def plot_residuals(values, predictions):
    plt.figure()
    (predictions - values).hist(bins=100)
    plt.xlabel('Residuals')
    plt.ylabel('Freq.')
    plt.axis([-10000,10000,0,5000])
    return plt
#plot_residuals(values, predictions)
#print (abs(predictions - values)).describe()