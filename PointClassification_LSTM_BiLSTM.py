#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries
import tensorflow
import numpy as np
from tensorflow.keras.utils import to_categorical
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report as cr 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn import metrics
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import keras
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#print(tensorflow.__version__)


# ### Loading and Preparing Data

# In[3]:


#preparing the dataset
import pandas as pd
import glob

all_files = glob.glob("*.csv")

li = []
s_id = 1
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    df['s_id'] = s_id
    li.append(df)
    s_id +=1

df = pd.concat(li, axis=0, ignore_index=True)

del df[0]

df.columns = ['x_acc', 'y_acc', 'z_acc', 'activity', 's_id']

activity = {
    1: 'Working at Computer',
 2: 'Standing Up, Walking and Going updown stairs',
 3: 'Standing',
 4: 'Walking',
 5: 'Going UpDown Stairs',
 6: 'Walking and Talking with Someone',
 7: 'Talking while Standing'
 }


df['activity'] = df['activity'].map(activity)

df.head()


# ### Cleaning the Data

# In[4]:


#summary statistics
df.iloc[:,0:3].describe()


# In[5]:


#checking for null values
print(df.isna().sum())


# In[6]:


#removing na values
df = df.dropna()


# ### Plotting the Data

# In[6]:


#counts of activities
df['activity'].hist(orientation="horizontal")


# In[7]:


#distribution of activity by person

df['activity'].hist(by=df['s_id'], orientation="horizontal", figsize = (15,10), sharey = True, sharex = True, layout = (3,5))


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(16,8))
sns.color_palette("tab10")
plt.title('Data on each subject', fontsize=24)
sns.countplot(x='s_id', data = df)
plt.xlabel("person", size=23)
plt.ylabel("Count", size=23)
plt.xticks(size=15)
plt.show()


# In[7]:


fig = px.pie(df, names='activity',width=980)
fig.update_layout(
    title={
        'text': "Activities distribution in the data",
        'y':0.95,
        'x':0.40,
        'xanchor': 'center',
        'yanchor': 'top'},
         legend_title ="Activities",
         font=dict(
         family="Arial",
         size=18))
fig.show()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.figure(figsize=(16,8))
sns.color_palette("tab10")
plt.title('Data on each subject by activity', fontsize=24)
sns.countplot(x='s_id',hue='activity', data = df)
plt.xlabel("person", size=23)
plt.ylabel("Count", size=23)
plt.xticks(size=15)
plt.show()


# In[120]:





# In[12]:


#sampling freq is 50Hz, let's study changes in x, y and z acceleration over time for the first subject during various activities
tmp = df[(df['s_id'] == 1) & (df['activity'] == 'Working at Computer')][['x_acc', 'y_acc', 'z_acc']]
tmp = tmp.melt(var_name='axis', value_name='acceleration')

plt.figure(figsize=(5,5))
sns.boxplot(x = 'axis', y='acceleration', data=tmp, showfliers=False)
#plt.axhline(y=0.08, xmin=0.1, xmax=0.9,c='m',dashes=(5,3))
plt.title('Ranges of Acceleration while Working on Computer', fontsize=25)
plt.xlabel("Activity", size=23)
plt.ylabel('Angle (X & Gravity)', size=22)
plt.xticks(rotation = 30, fontsize = 20)
plt.show()


# In[13]:


#sampling freq is 50Hz, let's study changes in x, y and z acceleration over time for the first subject during various activities
tmp = df[(df['s_id'] == 1) & (df['activity'] == 'Talking while Standing')][['x_acc', 'y_acc', 'z_acc']]
tmp = tmp.melt(var_name='axis', value_name='acceleration')

plt.figure(figsize=(5,5))
sns.boxplot(x = 'axis', y='acceleration', data=tmp, showfliers=False)
#plt.axhline(y=0.08, xmin=0.1, xmax=0.9,c='m',dashes=(5,3))
plt.title('Ranges of Acceleration while Talking while Standing', fontsize=25)
plt.xlabel("Activity", size=23)
plt.ylabel('Angle (X & Gravity)', size=22)
plt.xticks(rotation = 30, fontsize = 20)
plt.show()


# In[14]:


#sampling freq is 50Hz, let's study changes in x, y and z acceleration over time for the first subject during various activities
tmp = df[(df['s_id'] == 1) & (df['activity'] == 'Walking')][['x_acc', 'y_acc', 'z_acc']]
tmp = tmp.melt(var_name='axis', value_name='acceleration')

plt.figure(figsize=(5,5))
sns.boxplot(x = 'axis', y='acceleration', data=tmp, showfliers=False)
#plt.axhline(y=0.08, xmin=0.1, xmax=0.9,c='m',dashes=(5,3))
plt.title('Ranges of Acceleration while Walking', fontsize=25)
plt.xlabel("Activity", size=23)
plt.ylabel('Angle (X & Gravity)', size=22)
plt.xticks(rotation = 30, fontsize = 20)
plt.show()

While ranges of accelerations during different activities vary as can be observed, we will now observe changes in these activities over time
# In[13]:


import warnings
warnings.filterwarnings("ignore")


# In[14]:


tmp = df[(df['s_id'] == 1)] #& (df['activity'] == 'Talking while Standing')]#[['x_acc', 'y_acc', 'z_acc']]
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(tmp, hue='activity', aspect=2)
facetgrid.map(sns.distplot,'y_acc', hist=False)    .add_legend()
plt.title('Stationary vs Moving activities', fontsize=25)
plt.xlabel("Acc Magnitude mean", size=20)
plt.ylabel('Density', size=20)
plt.show()


# In[16]:


#difficult to interpret, let's visualize moving and stationary activities separately
import warnings
warnings.filterwarnings("ignore")
tmp = df[(df['s_id'] == 1) & (df['activity'].isin(['Standing Up, Walking and Going updown stairs', 'Walking', 'Walking and Talking with Someone']))]
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(tmp, hue='activity',aspect=2)
facetgrid.map(sns.distplot,'x_acc', hist=False)    .add_legend()
facetgrid.map(sns.distplot,'z_acc', hist=False)    .add_legend()
facetgrid.map(sns.distplot,'y_acc', hist=False)    .add_legend()

plt.annotate("x_acc", xy=(1900, 0.015), xytext=(1930, 0.0160), size=25,            va='center', ha='left',            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("z_acc", xy=(2060, 0.009), xytext=(2090, 0.009), size=25,            va='center', ha='left',            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.annotate("y_acc", xy=(2400, 0.010), xytext=(2430, 0.010), size=25,            va='center', ha='left',            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))


plt.title('Acceleration during Moving Activities', fontsize=25)
plt.xlabel("Acceleration", size=20)
plt.ylabel('Density', size=20)
plt.show()


# In[9]:


#difficult to interpret, let's visualize moving and stationary activities separately
import warnings
warnings.filterwarnings("ignore")
tmp = df[(df['s_id'] == 1) & (df['activity'].isin(['Working at Computer', 'Standing', 'Talking while Standing']))]
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(tmp, hue='activity', aspect=2)
facetgrid.map(sns.distplot,'x_acc', hist=False)    .add_legend()
facetgrid.map(sns.distplot,'z_acc', hist=False)    .add_legend()
facetgrid.map(sns.distplot,'y_acc', hist=False)    .add_legend()

plt.annotate("x_acc", xy=(1980, 0.04), xytext=(2030, 0.04), size=25,            va='center', ha='left',            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("z_acc", xy=(2140, 0.024), xytext=(2180, 0.029), size=25,            va='center', ha='left',            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.annotate("y_acc", xy=(2400, 0.06), xytext=(2430, 0.07), size=25,            va='center', ha='left',            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))


plt.title('Acceleration during Stationary Activities', fontsize=25)
plt.xlabel("Acceleration", size=20)
plt.ylabel('Density', size=20)
plt.show()

This gives a good estimate of the differences that exist between accellerations over different axes 
during different activities. 

This also explains why most point classification algorithms perform quite well. Let's what differences exist when seen
over time

# In[8]:


#adding a time variable to the activities

#df[(df['s_id'] == 1) & (df['activity'] == 'Going UpDown Stairs')]
g = df.groupby(['s_id', 'activity'], as_index=False)
df['time'] = g.cumcount()


# In[16]:


tmp = df[(df['s_id'] == 1) & (df['activity'].isin(['Working at Computer']))][['x_acc', 'y_acc', 'z_acc', 'time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Working at Computer')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()


# In[260]:


tmp = df[(df['s_id'] == 1) & (df['activity'].isin(['Going UpDown Stairs']))][['x_acc', 'y_acc', 'z_acc', 'time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Going UpDown Stairs')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()


# In[259]:


tmp = df[(df['s_id'] == 1) & (df['activity'].isin(['Walking and Talking with Someone']))][['x_acc', 'y_acc', 'z_acc', 'time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Walking and Talking with Someone')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()


# In[262]:


tmp = df[(df['s_id'] == 1) & (df['activity'].isin(['Talking while Standing']))][['x_acc', 'y_acc', 'z_acc', 'time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Talking while Standing')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()

Well defined differences can be seen among accelerations along all axes during different activities over time. This temporal
information may be extremely useful in modeling these activities
# ### Benchmark Model: Evaluating Point Classification Algorithms,   
A point classification algorithm is one that does not make use of the temporal information associated with the data. 
These models will be compared with time series classification algorithms. 

We will implement two point classificaton algorithms that have previously been implemented for this task, to obtain a benchmark that we will attempt to beat using a time series classification algorithm. 
# ### Train Test Split

# In[23]:


from sklearn.model_selection import train_test_split 
x = df.iloc[:, 0:3] #Features (Independent Variables)
y = df.iloc[:, -3] #Target Variables (Dependent Variables)
#creating a 75/25 train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25) 
print("X Train:\n", x_train.head(3), "\n")
print("Y Train:\n", y_train.head(3), "\n")
print("X Test :\n", x_test.head(3), "\n")
print("Y Test:\n", y_test.head(3), "\n")


# ### Implementing and Evaluating Decision Tree Classifiers

# In[27]:


from sklearn import metrics
from sklearn.metrics import accuracy_score as acc
from sklearn.tree import DecisionTreeClassifier as dtc

gini_acc = []
entropy_acc = []
val = 0
val2 = 0
print("\n Accuracy with Splitting based on Gini Impurity: \n")
for i in range(1, 20):
    k = i+1
    decisionTreeClassifier = dtc(criterion = 'gini', max_depth = k)
    decisionTreeClassifier.fit(x_train, y_train)
    y_predict = decisionTreeClassifier.predict(x_test)
    a = acc(y_test, y_predict)*100
    if a > val:
        val = a
        ind = k
    gini_acc.append(a)
    print("GINI Impurity based Splitting with Max_depth =", k, " is ", a, "%")

print("\n Accuracy with Splitting based on Gini Impurity: \n")
for i in range(1, 20):
    k = i+1
    decisionTreeClassifier = dtc(criterion = 'entropy', max_depth = k)
    decisionTreeClassifier.fit(x_train, y_train)
    y_predict = decisionTreeClassifier.predict(x_test)
    a = acc(y_test, y_predict)*100
    if a > val2:
        val2 = a
        ind = k
    entropy_acc.append(a)
    print("ENTROPY based splitting with Max_depth =", k, " is ", a, "%")
    
print("\n")


# In[31]:


#obtaining maximum accuracies:
print("Max Accuracy with Gini: ", max(gini_acc),', Depth = ', 2+gini_acc.index(max(gini_acc)) ,'\n')
print("Max Accuracy with Gini: ", max(entropy_acc),', Depth = ', 2+entropy_acc.index(max(entropy_acc)) ,'\n')

Maximum accurcy is obtained with Entropy based splitting, with a maximum tree depth of 15. 
# In[34]:



print("Criterion selected as ENTROPY and max depth", ind, "will give us an accuracy score of ", max(entropy_acc))
plt.figure(figsize=(16,5))
plt.title("Model Accuracy Score: \n")
plt.ylabel("Accuracy Scores: (in %)")
plt.ylim(40, 100)
plt.xlim(0, 25)
plt.xlabel("Max_depth")
plt.plot(range(1, 20), entropy_acc)
plt.vlines(ind, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors = 'red');


# In[36]:


#evaluation 


decision_tree = dtc(criterion = 'gini', max_depth = 15) #max accuracy decision tree
decision_tree.fit(x_train, y_train) 
y_predicted = decision_tree.predict(x_test)


# In[38]:


#Accuracy Score for the above trained decision tree classifier with default parameter
print('Accuracy Score for Normal Decision Tree Classifier: ', (metrics.accuracy_score(y_test, y_predicted)*100))
#Printing the classification report for the normal decision tree
print("Classification Report for Normal Decision Tree :")
target_names = ['Working at computer', 'Standing up, Walking and going updown', 'Standing', 'Walking', 'Going upDown stairs', 'Walking and talking', 'Talking While Standing']
print(cr(y_test, y_predicted, digits = 3, target_names=target_names))
#Confusion Matrix
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:")
print(cm, "\n")


# ### Implementing and Evaluating KNN Classifier

# In[41]:



accur = []
optimum = 0
val3 = 0
for i in range(1, 25):
    k = i+1
    neighbors = knc(n_neighbors = k)
    neighbors.fit(x_train, y_train)
    y_predict = neighbors.predict(x_test)
    a = acc(y_test, y_predict)*100
    accur.append(a)
    if a > val3:
        val3 = a
        optimum = k
    print("Accuracy: ", a, "% for ", k, 'nearest neighbours')

print("\n The optimum k for KNN = ", optimum)


plt.figure(figsize=(16,5))
plt.title("Model Accuracy Score: \n")
plt.ylabel("Accuracy Scores: (in %)")
plt.ylim(40, 100)
plt.xlim(0, 25)
plt.xlabel("Neighbors")
plt.plot(range(1, 25), accur)


# In[42]:


#evaluating on test set

knnModel = knc(25)
knnModel_1 = knnModel.fit(x_train, y_train) 
y_pred = knnModel_1.predict(x_test)


# In[43]:


print("Accuracy for KNN, k = 25: ", metrics.accuracy_score(y_test, y_pred))
print("Classification Report: \n", cr(y_test, y_pred))
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# ### Notes on Point Classification 
The highest obtained accuracy from point classification is 75.13 which is obtained from a KNN classification algorithm with k = 25. 
# ## Analyzing Temporal Patterns

# In[61]:


#x axis acceleration for subject_1 in activity 
tmp = df[(df['s_id'] == 6) & (df['activity'].isin(['Walking']))][['y_acc','time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Y-acceleration during Walking Subject 6')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()

#x axis acceleration for subject_2 in activity 
tmp = df[(df['s_id'] == 3) & (df['activity'].isin(['Walking']))][['y_acc','time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Y-acceleration during Walking, Subject 3')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()


# In[62]:


#x axis acceleration for subject_1 in activity 
tmp = df[(df['s_id'] == 6) & (df['activity'].isin(['Talking while Standing']))][['y_acc','time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Y-acceleration during Talking while Standing Subject 6')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()

#x axis acceleration for subject_2 in activity 
tmp = df[(df['s_id'] == 3) & (df['activity'].isin(['Talking while Standing']))][['y_acc','time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Y-acceleration during Talking while Standing, Subject 3')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()


# In[63]:


#x axis acceleration for subject_1 in activity 
tmp = df[(df['s_id'] == 2) & (df['activity'].isin(['Working at Computer']))][['z_acc','time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Z-acceleration during Working at Computer Subject 2')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()

#x axis acceleration for subject_2 in activity 
tmp = df[(df['s_id'] == 3) & (df['activity'].isin(['Working at Computer']))][['z_acc','time']]
tmp = tmp.melt(id_vars = ['time'], var_name='axis', value_name='acceleration')
tmp = tmp.set_index('time')
tmp.groupby('axis')['acceleration'].plot(legend='True')
plt.title('Z-acceleration during Working at Computer, Subject 3')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.show()


# # MultiClass Time Series Classification

# In[131]:




x = []
y = []
z = []
acc = []
s_id = [] 
t = []
id_ = []
step = 40 #length of overlap between time windows
window = 200 #length of a single time series observation
users = df['s_id'].unique()
activity = list(df['activity'].unique())

for k in users:
    for j in activity:
        data = df[(df['s_id'] == k) & (df['activity'] == j)]
        for i in range (len(data) - step):
            if(i >= int(len(data)/step)):
                break
            if(len(list(data['x_acc'].iloc[(i*step):(i*step)+window])) < window):
                break
            x.append(list(data['x_acc'].iloc[(i*step):(i*step)+window]))
            y.append(list((data['y_acc'].iloc[(i*step):(i*step)+window])))
            z.append(list((data['z_acc'].iloc[(i*step):(i*step)+window])))
            acc.append(data['activity'].iloc[(i*step):(i*step)+window].unique()[0])
            s_id.append(k)
            id_.append(list(data['id'].iloc[(i*step):(i*step)+window]))
            t.append(list((data['time'].iloc[(i*step):(i*step)+window])))


# In[132]:


#merge x, y and z acceleration windows to create x data
X_data = []
X_data.append(x)
X_data.append(y)
X_data.append(z)
X_data = np.dstack(X_data)


# In[133]:


#shape of X data
X_data.shape


# In[134]:


#mapping activity to numerical values
activity = {
    1: 'Working at Computer',
 2: 'Standing Up, Walking and Going updown stairs',
 3: 'Standing',
 4: 'Walking',
 5: 'Going UpDown Stairs',
 6: 'Walking and Talking with Someone',
 7: 'Talking while Standing'
 }

activity_inv = {v: k for k, v in activity.items()}
acc_map = [activity_inv[k] for k in acc]


# In[135]:


#create one hot encoded categories with keras to_categorical 

Y_data = to_categorical(acc_map)


# In[136]:


#create train test splits
from sklearn.model_selection import train_test_split
#train test split by index
all_indices = list(range(len(x)))
train_index, test_index = train_test_split(all_indices, test_size = 0.3)


#splits based on indices extracted
trainX = X_data[train_index,:]
testX = X_data[test_index, :]

trainy = Y_data[train_index,:]
testy = Y_data[test_index, :]


# In[41]:


#check to verify shapes of train and test sets


# In[137]:


trainX.shape


# In[138]:


testX.shape


# In[139]:


trainy.shape


# In[140]:


testy.shape


# In[46]:


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 15, 100
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy


# In[47]:


scores = list()
repeats = 10
for r in range(repeats):
    score = evaluate_model(trainX, trainy, testX, testy)
    score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)


# In[151]:


#
df_f = df

df_f['x_acc'] = df['x_acc'].astype(float)
df_f['y_acc'] = df['y_acc'].astype(float)
df_f['z_acc'] = df['z_acc'].astype(float)


# In[152]:


x = []
y = []
z = []
acc = []
s_id = [] 
t = []
id_ = []
step = 40 #length of overlap between time windows
window = 200
users = df['s_id'].unique()
activity = list(df['activity'].unique())

for k in users:
    for j in activity:
        data = df_f[(df_f['s_id'] == k) & (df_f['activity'] == j)]
        for i in range (len(data) - step):
            if(i >= int(len(data)/step)):
                break
            if(len(list(data['x_acc'].iloc[(i*step):(i*step)+window])) < window):
                break
            x.append(list(data['x_acc'].iloc[(i*step):(i*step)+window]))
            y.append(list((data['y_acc'].iloc[(i*step):(i*step)+window])))
            z.append(list((data['z_acc'].iloc[(i*step):(i*step)+window])))
            acc.append(data['activity'].iloc[(i*step):(i*step)+window].unique()[0])
            s_id.append(k)
            id_.append(list(data['id'].iloc[(i*step):(i*step)+window]))
            t.append(list((data['time'].iloc[(i*step):(i*step)+window])))


# In[153]:


#merge x, y and z acceleration windows to create x data
X_data = []
X_data.append(x)
X_data.append(y)
X_data.append(z)
X_data = np.dstack(X_data)


# In[ ]:


#mapping activity to numerical values
activity = {
 1: 'Working at Computer',
 2: 'Standing Up, Walking and Going updown stairs',
 3: 'Standing',
 4: 'Walking',
 5: 'Going UpDown Stairs',
 6: 'Walking and Talking with Someone',
 7: 'Talking while Standing'
 }

activity_inv = {v: k for k, v in activity.items()}
acc_map = [activity_inv[k] for k in acc]


# In[155]:


Y_data = acc


# In[168]:


#create one hot encoded categories with keras to_categorical 

Y_data = to_categorical(acc_map)


# In[169]:


#create train test splits
from sklearn.model_selection import train_test_split
#train test split by index
all_indices = list(range(len(x)))
train_index, test_index = train_test_split(all_indices, test_size = 0.3)


#splits based on indices extracted
trainX = X_data[train_index,:]
testX = X_data[test_index, :]

trainy = Y_data[train_index,:]
testy = Y_data[test_index, :]


# In[17]:


model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=128,
          input_shape=[trainX.shape[1], trainX.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(trainy.shape[1], activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)


# In[171]:


history = model.fit(
    trainX, trainy,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False)


# In[172]:


#with scaling


# In[12]:


df_train = df[df['s_id'] <= 12]
df_test = df[df['s_id'] > 12]


# In[13]:


scale_columns = ['x_acc', 'y_acc', 'z_acc']

scaler = RobustScaler()

scaler = scaler.fit(df_train[scale_columns])

df_train.loc[:, scale_columns] = scaler.transform(
  df_train[scale_columns].to_numpy()
)

df_test.loc[:, scale_columns] = scaler.transform(
  df_test[scale_columns].to_numpy()
)


# In[18]:


def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


# In[14]:


def create_dataset(X, y, time_steps, step):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)


# In[16]:


TIME_STEPS = 25
STEP = 25

X_train, y_train = create_dataset(
    df_train[['x_acc', 'y_acc', 'z_acc']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[['x_acc', 'y_acc', 'z_acc']],
    df_test.activity,
    TIME_STEPS,
    STEP
)


# In[18]:


print(X_train.shape, y_train.shape)


# In[18]:


enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)


# In[19]:


import keras
model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
      keras.layers.LSTM(
          units=128,
          input_shape=[X_train.shape[1], X_train.shape[2]]
      )
    )
)
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['acc']
)


# In[22]:


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


# In[23]:


TIME_STEPS = 15
STEP = 10

X_train, y_train = create_dataset(
    df_train[['x_acc', 'y_acc', 'z_acc']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[['x_acc', 'y_acc', 'z_acc']],
    df_test.activity,
    TIME_STEPS,
    STEP
)


# In[26]:


print(X_train.shape, y_train.shape)


# In[27]:


enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)


# In[28]:


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


# In[70]:


plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# Print confusion matrix for training data
y_pred_train = model.predict(X_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(np.argmax(y_train, axis = 1), max_y_pred_train))


# In[75]:


print("Accuracy for Bi LSTM: ", metrics.accuracy_score(np.argmax(y_train, axis = 1), max_y_pred_train))
print("Classification Report: \n", cr(np.argmax(y_train, axis = 1), max_y_pred_train))
#Confusion Matrix
cm = confusion_matrix(np.argmax(y_train, axis = 1), max_y_pred_train)
print("Confusion Matrix:\n", cm)


# In[76]:


TIME_STEPS = 3
STEP = 3

X_train, y_train = create_dataset(
    df_train[['x_acc', 'y_acc', 'z_acc']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[['x_acc', 'y_acc', 'z_acc']],
    df_test.activity,
    TIME_STEPS,
    STEP
)


# In[77]:


print(X_train.shape, y_train.shape)


# In[78]:


enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)


# In[80]:


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size= 10000,
    validation_split=0.1,
    shuffle=False
)


# In[81]:


plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# Print confusion matrix for training data
y_pred_train = model.predict(X_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(np.argmax(y_train, axis = 1), max_y_pred_train))


# In[82]:


TIME_STEPS = 8
STEP = 5

X_train, y_train = create_dataset(
    df_train[['x_acc', 'y_acc', 'z_acc']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[['x_acc', 'y_acc', 'z_acc']],
    df_test.activity,
    TIME_STEPS,
    STEP
)

print(X_train.shape, y_train.shape)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size= 1000,
    validation_split=0.1,
    shuffle=False
)


# In[83]:


plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# Print confusion matrix for training data
y_pred_train = model.predict(X_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(np.argmax(y_train, axis = 1), max_y_pred_train))


# In[84]:


TIME_STEPS = 12
STEP = 5

X_train, y_train = create_dataset(
    df_train[['x_acc', 'y_acc', 'z_acc']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[['x_acc', 'y_acc', 'z_acc']],
    df_test.activity,
    TIME_STEPS,
    STEP
)

print(X_train.shape, y_train.shape)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

history = model.fit(
    X_train, y_train,
    epochs=8,
    batch_size= 1000,
    validation_split=0.1,
    shuffle=False
)


# In[85]:


plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# Print confusion matrix for training data
y_pred_train = model.predict(X_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(np.argmax(y_train, axis = 1), max_y_pred_train))


# In[23]:


TIME_STEPS = 12
STEP = 5

X_train, y_train = create_dataset(
    df_train[['x_acc', 'y_acc', 'z_acc']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[['x_acc', 'y_acc', 'z_acc']],
    df_test.activity,
    TIME_STEPS,
    STEP
)

print(X_train.shape, y_train.shape)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size= 1500,
    validation_split=0.1,
    shuffle=False
)


# In[24]:


plt.figure(figsize=(6, 4))
plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# Print confusion matrix for training data
y_pred_train = model.predict(X_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(np.argmax(y_train, axis = 1), max_y_pred_train))

