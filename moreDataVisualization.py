# -*- coding: utf-8 -*-
import pandas as pd
from ggplot import *

file1 = 'C:\Users\Daniel\Downloads\improved-dataset\\turnstile_weather_v2.csv'
file2 = 'C:\Users\Daniel\Downloads\\turnstile_data_master_with_weather.csv'
turnstile_weather_v2 = pd.read_csv(file1)

def plot_weather_data(turnstile_weather_v2):
    '''
    plot the histogram of ENTRIESn_hourly over day_week
    '''
    plot = ggplot(turnstile_weather_v2, aes('day_week', 'ENTRIESn_hourly')) + \
    geom_histogram() + \
    ggtitle('Histogram of ENTRIESn_hourly over day_week') + \
    xlab('day_week') + ylab('ENTRIESn_hourly') +\
    xlim(0, 6)
    return plot
#print plot_weather_data(turnstile_weather_v2)

for c in turnstile_weather_v2.columns.values:
    print ggplot(turnstile_weather_v2, aes(c, 'ENTRIESn_hourly')) + \
    geom_histogram() + \
    ggtitle('Histogram of ENTRIESn_hourly') + \
    xlab(c) + ylab('ENTRIESn_hourly')