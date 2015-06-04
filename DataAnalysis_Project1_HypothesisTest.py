# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file. 
"""

import pandas as pd
import numpy as np
#import pandasql
import scipy
import scipy.stats

from ggplot import *
import matplotlib.pyplot as plt


file1 = 'C:\Users\Daniel\Downloads\improved-dataset\\turnstile_weather_v2.csv'
file2 = 'C:\Users\Daniel\Downloads\\turnstile_data_master_with_weather.csv'
turnstile_weather_v2 = pd.read_csv(file1)

# scipy.stats.mannwhitneyu returns p-value as nan, but returns it normally when several rows of data set are discarded
turnstile_weather_v2 = turnstile_weather_v2[6:len(turnstile_weather_v2['rain'])]


turnstile_noRain = turnstile_weather_v2[turnstile_weather_v2['rain'] == 0]
turnstile_rain = turnstile_weather_v2[turnstile_weather_v2['rain'] == 1]
entry_noRain = turnstile_noRain['ENTRIESn_hourly']
entry_rain = turnstile_rain['ENTRIESn_hourly']


### Data Wangling
# print turnstile_weather_v2.describe()
# print turnstile_weather_v2.head()
# print np.where(pd.isnull(turnstile_weather_v2))

### Data Visualization
# Histogram of ENTRIESn_hourly
def plot_ENTRIESn_hourly_hist(entry_noRain, entry_rain):
    plt.figure(figsize=(10,8))
    entry_noRain.plot(color='blue', kind='hist', bins=100, label='No Rain')
    entry_rain.plot(color='green', kind='hist', bins=100, label='Rain')
    plt.axis([0,10000,0,10000])
    #plt.axis([0,5e7,0,400])
    plt.xlabel('ENTRIESn_hourly')
    plt.ylabel('Freq.')
    plt.title('Histogram of ENTRIESn_hourly')
    plt.legend()
plot_ENTRIESn_hourly_hist(entry_noRain, entry_rain)
    


'''
print ggplot(turnstile_noRain, aes('Hour', 'ENTRIESn_hourly')) + \
    geom_point(color='blue') + geom_line() + \
    ggtitle('Histogram of ENTRIESn_hourly') + xlab('ENTRIESn_hourly') + ylab('Freq.') 
'''


### Data Analyzing
'''
From above figure, the samples do not follow normal distrubution, so T-test is not
applicable, and thus I decide to use Mann-Whitney U test which does not assume
normal distibution.
'''
entry_noRain_mean = np.mean(entry_noRain)
entry_rain_mean = np.mean(entry_rain)
print "The means (rainy & non-rainy) are:", entry_rain_mean, entry_noRain_mean
U,p_oneTail = scipy.stats.mannwhitneyu(entry_rain.values, entry_noRain.values)
print "scipy.stats.mannwhitneyu returns U:", U
print "scipy.stats.mannwhitneyu returns one-tail p-value:", p_oneTail
p_twoTail = p_oneTail * 2
print "Two-tail p-value is:", p_twoTail
alpha = 0.05
if p_twoTail < alpha:
    print "Reject the null hypothesis since the two-tail p-value < alpha =", alpha
elif p_twoTail > alpha:
    print "Fail to reject the null hypothesis since the two-tail p-value >= alpha =", alpha
else:
    print "Error when comparing p_twoTail with alpha!"

