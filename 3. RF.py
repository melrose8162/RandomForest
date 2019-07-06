import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import export_graphviz  
import pydot
import matplotlib.pyplot as plt 
import datetime


# Load Dataset - print head and shape 
df = pd.read_csv('WA.csv')
pd.set_option('display.max_columns', 13)
#print(df.head(7))
#print(df.shape)

# See Summary Statistics
#print(df.describe())
#print(list(df))

# One Hot Encoding - convert DOW to num values
df = pd.get_dummies(df)
print(df.head(5))

# Display first 5 rows of the last 12 columns
#print(df.iloc[:, 5:].head(5))
#print(df.shape)

# Pandas to Numpy
print(list(df))

# Labels = dependent variable, features = independent variables 
# first split dep and indep, then split train and test ********
labels = np.array(df['actual'])
features = df.drop('actual', axis=1)
features_list = list(features.columns)
features = np.array(features)

# Random state = 42, for reproducible results. Split into test and train sets 
train_features, test_features, train_labels, test_labels =train_test_split(features, labels, test_size=1/4, random_state=42)
print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

# Establish baseline error rate - essentially average of max_temps in the test set as it is indeed
# max temps we are trying to predict - then estimate errors by diff (avg max temps and test set)

# Instatiate the model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

# Predict on test data - using forest method's predict data 
# predict on test features 
predict = rf.predict(test_features)
# Calculate absolute errors
abs_error =  abs(predict - test_labels)
# print mean absolute errors
print('Mean Abs Error: ', round(np.mean(abs_error), 2), 'degrees')

# Estimate Mean Abs PCT Error and Acuracy
print('Mean Abs PCT Error:', round(np.mean(abs_error/test_labels), 2)*100, 'degrees')
Accuracy = (100  - round((np.mean(abs_error/test_labels)*100), 1))
print('Accuracy:', Accuracy, 'degrees')

# Tree 
tree = rf.estimators_[5] #pull out one tree from the forest 
export_graphviz(tree, out_file = 'tree.dot', feature_names = features_list, rounded = True, precision=1)
# Export image to a dot file 

(graph, ) = pydot.graph_from_dot_file('tree.dot')
# use dot file to create a graph

#graph.write_png('tree.png')
#write graph to png file 

# Variable Importance Plot 
importances = list(rf.feature_importances_)
# List of tupes with variables and importance
feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
# Sort by most importance features first
feature_importance = sorted(feature_importance, key = lambda x: x[1], reverse=True)
[print
('Variable:  {:20} Importances: {}'.format(*pair)) for pair in feature_importance
]

# Plot VAR IMP
plt.style.use('fivethirtyeight')
plt.barh(features_list, importances)
plt.title('Variable Important Plot')
plt.xlabel('VAR IMP')
plt.ylabel('Features')
plt.show()

# Plot Predicted and Actual 

# Dates for Training Values
mnth = features[:, features_list.index('month')]
day = features[:, features_list.index('day')]
year = features[:, features_list.index('year')]
dates = [str(int(year))+ '-' + str(int(mnth))+ '-' + str(int(day)) for year, mnth, day in zip(year, mnth, day)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# Dates for Test Values 
mnth = test_features[:, features_list.index('month')]
day = test_features[:, features_list.index('day')]
year = test_features[:, features_list.index('year')]
test_dates = [str(int(year))+ '-' + str(int(mnth))+ '-' + str(int(day)) for year, mnth, day in zip(year, mnth, day)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]


# DataFrame with Prediction and Dates
pred_data = pd.DataFrame(data = {'date' : test_dates, 'predictions' : predict})
# DataFrame with Actual Values and Dates
actual_data = pd.DataFrame(data = {'date': dates, 'actual' : labels})

plt.plot(pred_data['date'], pred_data['predictions'], 'bo', linewidth=3/2, label = 'predicted')
plt.plot(actual_data['date'], actual_data['actual'], 'r-', linewidth=3/2, label = 'actual')
plt.legend()
plt.show()







