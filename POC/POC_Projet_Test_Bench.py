#!/usr/bin/env python
# coding: utf-8

# # Concaténation des fichiers Result et TimeSeries

# In[1]:


import pandas as pd
import seaborn as sns


# In[2]:


df_result= pd.read_csv('/home/fitec/Mise en situation professionnelle/Projets FITEC/Test Bench/Results/Results_201210.csv', sep =';')


# In[3]:


df_result[:10]


# In[4]:


sns.countplot(df_result['RESULT'])


# In[5]:


from glob import glob

# csvs will contain all CSV files names ends with .csv in a list
csvs = glob('/home/fitec/Mise en situation professionnelle/Projets FITEC/Test Bench/TimeSeries/*.csv')

# remove the trailing .csv from CSV files names
#new_table_list = [csv[78:-4] for csv in csvs]
csv_path_list = [csv for csv in csvs]
filename_list = [string.split('_')[-1] for string in csv_path_list]
new_table_list = [string.split('.')[0] for string in filename_list]


# In[6]:


new_table_list


# In[7]:


len(new_table_list)


# In[8]:


import pandas as pd
TimeSeries_csv = pd.read_csv(csv_path_list[0], sep=';')
#TimeSeries_csv


# In[9]:


def get_values(dictionnary , serial_number, TimeSeries_file):
    dictionnary['SERIAL_NUMBER'].append(serial_number)
    dictionnary['SENSOR_1'].append(TimeSeries_file['SENSOR_1'].max())
    dictionnary['SENSOR_2'].append(TimeSeries_file['SENSOR_2'].max())
    dictionnary['SENSOR_3'].append(TimeSeries_file['SENSOR_3'].mean())
    dictionnary['SENSOR_4'].append(TimeSeries_file['SENSOR_4'].mean())
    dictionnary['SENSOR_5'].append(TimeSeries_file['SENSOR_5'].mean())
    dictionnary['PR_1'].append(TimeSeries_file['PR_1'].mean())
    dictionnary['PR_2'].append(TimeSeries_file['PR_2'].mean())
    dictionnary['PR_3'].append(TimeSeries_file['PR_3'].max())
    dictionnary['PR_4'].append(TimeSeries_file['PR_4'].max())
    dictionnary['TEMP_1'].append(TimeSeries_file['TEMP_1'].mean())
    dictionnary['TEMP_2'].append(TimeSeries_file['TEMP_2'].mean())
    dictionnary['BRAKE_1'].append(TimeSeries_file['BRAKE_1'].max())

d = {'SERIAL_NUMBER' : [],
     'SENSOR_1': [], 
     'SENSOR_2': [],
     'SENSOR_3': [], 
     'SENSOR_4': [],
     'SENSOR_5': [], 
     'PR_1': [],
     'PR_2': [],
     'PR_3': [],
     'PR_4': [],
     'TEMP_1': [],
     'TEMP_2': [],
     'BRAKE_1':[]}

for i in range(len(new_table_list)):
    serial_num = new_table_list[i]
    TimeSeries_csv = pd.read_csv(csv_path_list[i], sep=';')
    get_values(d, serial_num, TimeSeries_csv)


# In[10]:


import pandas as pd
df_TS = pd.DataFrame(data=d)
df_TS


# In[11]:


#df_TS.loc[df_TS['SERIAL_NUMBER'] == "6708c44794869fecc841594d0baa8046"]


# In[12]:


df_result_TS = pd.merge(df_result, df_TS, how="inner", on="SERIAL_NUMBER")
df_result_TS


# In[13]:


#df_result_TS.loc[df_result_TS['SERIAL_NUMBER'] == "6708c44794869fecc841594d0baa8046"]


# # Préparation d'un DataFrame pour le Machine Learning

# In[14]:


final_df = df_result_TS.drop(columns=['TEST_STAND', 'DATE', 'HEURE', 'SERIAL_NUMBER', 'PRODUCT_NUMBER','DIRECTION'])
final_df


# In[15]:


# Remplacer W par 0, et G par 1
final_df.replace(to_replace='W', value=0, inplace=True)
final_df.replace(to_replace='G', value=1, inplace=True)
final_df


# ## Séparer labels et features

# In[16]:


# Features
X = final_df.drop(columns='RESULT')
# Labels
y = final_df['RESULT']


# ## Uncersampling and Oversampling

# In[17]:


# Installer la bibliothèque à partir du terminal : sudo pip install imbalanced-learn
# Installer pour Jupyter Notebook : conda install -c conda-forge imbalanced-learn

# check version number
import imblearn
print(imblearn.__version__)


# In[18]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
print(Counter(y_over))


# # Machine Learning

# In[19]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sc = StandardScaler()
X_scaled = sc.fit_transform(X_over)

pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Training set et test set
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_over, test_size=0.33, random_state=42)

# Logistic Regression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)


# In[21]:


y_pred = clf.predict(X_test)
#print(f'y_pred = \n{y_pred}')
y_test_list = list(y_test)
#print(f'y_test = \n{y_test_list}')

# Calcul de l'accuracy à la main
#good_prediction = 0
#for i in range(len(y_pred)):
#    if y_test_list[i] == y_pred[i]:
#        good_prediction += 1
#score = good_prediction/len(y_pred)
#print(score)


# In[22]:


# Accuracy
clf.score(X_test,y_test)


# In[23]:


confusion_matrix = pd.crosstab(y_test, y_pred, rownames =['Classe réelle'], colnames=['Classe prédite'])
confusion_matrix


# In[24]:


import pickle
pickle.dump(clf, open('lgr_classifier_model.pkl', 'wb'))
print('Logistic Regression model saved as lgr_classifier_model.pkl')
