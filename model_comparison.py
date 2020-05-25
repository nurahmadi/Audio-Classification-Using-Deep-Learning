# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:51:59 2020

@author: na5815
"""
import os
import pandas as pd
from plotting import plot_model_comparison

result_dir = 'results'
features = ['mel','mfcc']
model_types = ['cnn','lstm']

f1_scores = []
labels = []
for feature in features:
    for model_type in model_types:
        report_file = os.path.join(result_dir,feature+'_'+model_type+'_test.csv')
        report = pd.read_csv(report_file,index_col=0)
        f1_scores.append(report['f1-score']['macro avg'])
        labels.append(feature+'_'+model_type)

fig, ax = plot_model_comparison(f1_scores,labels)

figure_file = os.path.join(result_dir,'model_comparison.png')
print ("Storing figure into a file: "+figure_file)
fig.savefig(figure_file, bbox_inches='tight')