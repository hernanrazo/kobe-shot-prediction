import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data onto a dataframe
df = pd.read_csv('/Users/hernanrazo/pythonProjects/kobe-shot-prediction/data.csv')

#set folder path for graphs
graph_folder_path = '/Users/hernanrazo/pythonProjects/kobe-shot-prediction/graphs/'

#check for empty values
print(df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')


#make frequency table of all action types
print(df.groupby('action_type').size())
print(' ')

#take the 20 least popular action types and combine them into one category of 'Other'
rare_action_types = df['action_type'].value_counts().sort_values().index.values[:20]
df.loc[df['action_type'].isin(rare_action_types), 'action_type'] = 'Other'

#turn action_type into a numerical variable by assigning each action a number
action_type_key = {'Alley Oop Dunk Shot':0, 'Alley Oop Layup shot':2,
'Driving Dunk Shot':3, 'Driving Finger Roll Layup Shot':4, 
'Driving Finger Roll Shot':5, 'Driving Jump shot':6, 'Driving Layup Shot':7,
'Driving Reverse Layup Shot':8, 'Driving Slam Dunk Shot':9, 'Dunk Shot':10,
'Fadeaway Bank shot':11, 'Fadeaway Jump Shot':12, 'Finger Roll Layup Shot':13,
'Finger Roll Shot':14, 'Floating Jump shot':15, 'Follow Up Dunk Shot':16, 
'Hook Shot':17, 'Jump Bank Shot':18, 'Jump Hook Shot':19, 'Jump Shot':20, 
'Layup Shot':21, 'Other':22, 'Pullup Jump shot':23, 'Putback Layup Shot':24, 
'Reverse Dunk Shot':25, 'Reverse Layup Shot':26, 'Reverse Slam Dunk Shot':27, 
'Running Bank shot':28, 'Running Dunk Shot':29, 'Running Hook Shot':30, 
'Running Jump Shot':31, 'Running Layup Shot':32, 'Slam Dunk Shot':33, 
'Step Back Jump shot':34, 'Tip Shot':35, 'Turnaround Bank shot':36, 
'Turnaround Fadeaway shot':37, 'Turnaround Jump Shot':38}

df['action_type'] = df['action_type'].map(action_type_key).astype(int)


#print a countplot for each category in the action_type variable
action_type_countplot = plt.figure()
plt.title('Occurance of Each Action Type')
sns.countplot(x = 'action_type', data = df)
action_type_countplot.savefig(graph_folder_path + 'action_type_barplot.png')

