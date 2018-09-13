import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

#load data onto a dataframe
df = pd.read_csv('/Users/hernanrazo/pythonProjects/kobe-shot-prediction/data.csv')

#set folder path for graphs
graph_folder_path = '/Users/hernanrazo/pythonProjects/kobe-shot-prediction/graphs/'

#check for empty values
print(df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')

#start off by cleaning the data 

#remove the obviously useless variables
df = df.drop(['game_id'], axis = 1)
df = df.drop(['team_name'], axis = 1)
df = df.drop(['matchup'], axis = 1)
df = df.drop(['shot_id'], axis = 1)
df = df.drop(['playoffs'], axis = 1)

#make frequency table of all action types
print(df.groupby('action_type').size())
print(' ')

#take the 20 least popular action types and combine them into one category called 'Other'
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


#do the same to the combine_shot_type variable
combined_shot_type_key = {'Bank Shot':0, 'Dunk':1, 'Hook Shot':2, 'Jump Shot':3,
'Layup':4, 'Tip Shot':5}

df['combined_shot_type'] = df['combined_shot_type'].map(combined_shot_type_key).astype(int)

#switch the shot_type variable into numeric
shot_type_key = {'2PT Field Goal':0, '3PT Field Goal':1}

df['shot_type'] = df['shot_type'].map(shot_type_key).astype(int)

#convert shot_zone_key to numeric 
shot_zone_area_key = {'Back Court(BC)':0, 'Center(C)':1, 'Left Side Center(LC)':2,
'Left Side(L)':3, 'Right Side Center(RC)':4, 'Right Side(R)':5}

df['shot_zone_area'] = df['shot_zone_area'].map(shot_zone_area_key).astype(int)

#convert the shot_zone_basic variable into numeric
shot_zone_basic_key = {'Above the Break 3':0, 'Backcourt':1, 'In The Paint (Non-RA)':2,
'Left Corner 3':3, 'Mid-Range':4, 'Restricted Area':5, 'Right Corner 3':6}

df['shot_zone_basic'] = df['shot_zone_basic'].map(shot_zone_basic_key).astype(int)

#turn shot_zone_range into numeric
shot_zone_range_key = {'16-24 ft.':0, '24+ ft.':1, '8-16 ft.':2, 'Back Court Shot':3,
'Less Than 8 ft.':4}

df['shot_zone_range'] = df['shot_zone_range'].map(shot_zone_range_key).astype(int)


#combine the seconds_remaining and minutes_remaining variables by converting minutes
#to seconds and then adding them
df['total_sec_left'] = df['minutes_remaining'] * 60 + df['seconds_remaining'] 

#drop the original variables
#df.drop(['minutes_remaining'], axis = 1)
#df.drop(['seconds_remaining'], axis = 1)

#convert the game_date variable into a dateframe and then seperate it into 
#2 new variables of year and month
df['game_date'] = pd.to_datetime(df['game_date'])
df['game_year'] = df['game_date'].dt.year
df['game_month'] = df['game_date'].dt.month

#drop original game_date variable
df = df.drop(['game_date'], axis = 1)

#start visualizing data

#make simple plots of shot coordinates
shot_coordinate_plot = plt.figure(figsize = (7, 7*(84.0/50.0)))
plt.title('Shot x & y')
sns.regplot(x = 'loc_x', y ='loc_y', data = df)
shot_coordinate_plot.savefig(graph_folder_path + 'shot_coordinate_plot.png')

shot_lat_lon_plot = plt.figure(figsize = (7, 7*(84.0/50.0)))
plt.title('Shot Lan & Lon')
sns.regplot(x = 'lat', y = 'lon', data = df)
shot_lat_lon_plot.savefig(graph_folder_path + 'shot_lat_lon_plot.png')

#print scatter plots of shots made relative to how many seconds and minutes
#were left in the game
shots_sec_plot = plt.figure(figsize = (7, 7*(84.0/50.0)))
plt.title('Shots When Seconds Remain')
plt.scatter(df.loc_x, df.loc_y, alpha = 0.7, c = df.seconds_remaining,
	cmap = 'Greens_r')

plt.savefig(graph_folder_path + 'shots_sec_plot.png')

shots_min_plot = plt.figure(figsize = (7, 7*(84.0/50.0)))
plt.title('Shots when minutes Remain')
plt.scatter(df.loc_x, df.loc_y, alpha = 0.7, c = df.minutes_remaining,
	cmap = 'Greens_r')
plt.savefig(graph_folder_path + 'shots_min_plot.png')

#print a countplot for each category in the action_type variable
action_type_countplot = plt.figure()
plt.title('Occurance of Each Action Type')
sns.countplot(x = 'action_type', data = df)
action_type_countplot.savefig(graph_folder_path + 'action_type_countplot.png')

#print countplot for the combined_shot variable
combined_shot_plot = plt.figure()
plt.title('Occurance of Each Combined Shot')
sns.countplot(x = 'combined_shot_type', data = df)
combined_shot_plot.savefig(graph_folder_path + 'combined_shot_plot.png')

#print counterplot for the shot_zone_area variable
shot_zone_area_plot = plt.figure()
plt.title('Shot Occurance on each Court Side')
sns.countplot(x = 'shot_zone_area', data = df)
shot_zone_area_plot.savefig(graph_folder_path + 'shot_zone_area_plot.png')

print(list(df))

#split dtat into a training section and a testing section
train_df, test_df = train_test_split(df, test_size = 0.3)

#take data features and output for training and testing
train_x = train_df['action_type', 'combined_shot_type', 'game_event_id', 'lat', 'loc_x', 'loc_y', 
'lon', 'minutes_remaining', 'period', 'season', 'seconds_remaining', 'shot_distance', 'shot_type', 
'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'opponent', 'total_sec_left', 
'game_year', 'game_month']

train_y = train_df['shot_made_flag']

test_x = test_df['action_type', 'combined_shot_type', 'game_event_id', 'lat', 'loc_x', 'loc_y', 
'lon', 'minutes_remaining', 'period', 'season', 'seconds_remaining', 'shot_distance', 'shot_type', 
'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'team_id', 'opponent', 'total_sec_left', 
'game_year', 'game_month']

test_y = test_df['shot_made_flag']

#double check everything came out correctly
print(train_df.head)
print(' ')
print(test_df.head)
print(' ')








