
# coding: utf-8

# In[304]:


#Import needed packages.
import numpy as np
import pandas as pd
import keras as k 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


# In[305]:


#Data later then 2021
#Only 'ANSI' encoder can decode blank at 0:0.
d = pd.read_csv('D://python/exam data0406(2).csv', encoding='ANSI')


# In[306]:


d1 = d[['PURID', 'PURPRW', 'DLSTK', 'SPECID', 'ORDCAQQTY', 'CPRUP', 'MKDYS', 'CAQUN', 'CUCY', 'CNTRNO', 'ORGI', 'WWWMK', 'ORDVND', 'ANS', 'COTYPE', 'BGUP', 'BGCUCY', 'EMGPURMK']]


# In[307]:


d1.head()


# In[308]:


d1['BG'] = d1['BGUP'][d1['BGCUCY'] == d1['CUCY']]
d1['BG'] = d1['BG'].fillna(0)
d1 = d1.drop(['BGUP', 'BGCUCY'], axis=1)


# In[309]:


clf = XGBClassifier()


# In[310]:


clf.load_model('xgboost1.h5')


# In[311]:


d2 = d1[['ORDCAQQTY', 'CPRUP', 'MKDYS']]


# In[312]:


apnd1 = clf.predict(d2)


# In[313]:


d3 = d1.copy()
d3['ANS2'] = apnd1


# In[314]:


#load original dic and use replace() to do as LabelEncoder.
d3['PURID'] = d3['PURID'].astype('str')
puriddic = np.load('puriddic.npy')
dic1 = dict(zip(puriddic, np.arange(len(puriddic))))
d3['PURID'].replace(dic1, inplace=True)


# In[315]:


#purprw
#d3['PURPRW'] = d3['PURPRW'].astype('str')
purprwdic = np.load('purprwdic.npy', allow_pickle=True)
d3['PURPRW'][~d3['PURPRW'].isin(purprwdic)] = ''
dic2 = dict(zip(purprwdic, np.arange(len(purprwdic))))
d3['PURPRW'].replace(dic2, inplace=True)
d3['PURPRW'].replace('', np.nan, inplace=True)


# In[316]:


#dlstk
#d3["DLSTK"] = d3["DLSTK"].astype('str')
dlstkdic = np.load('dlstkdic.npy', allow_pickle=True)
d3['DLSTK'][~d3['DLSTK'].isin(dlstkdic)] = ''
dic3 = dict(zip(dlstkdic, np.arange(len(dlstkdic))))
d3['DLSTK'].replace(dic3, inplace=True)
d3['DLSTK'].replace('', np.nan, inplace=True)


# In[317]:


#emgpurmk
d3['EMGPURMK'][d3['EMGPURMK']!='*'] = '0'
d3['EMGPURMK'][d3['EMGPURMK']=='*'] = '1'


# In[318]:


#specid
d3['SPECID'] = d3['SPECID'].str.slice(stop=2)
d3['SPECID'] = d3['SPECID'].astype('str')
speciddic = np.load('speciddic.npy', allow_pickle=True)
dic4 = dict(zip(speciddic, np.arange(len(speciddic))))
d3['SPECID'].replace(dic4, inplace=True)


# In[319]:


#caqun
d3['CAQUN'] = d3['CAQUN'].astype('str')
caqundic = np.load('caqundic.npy', allow_pickle=True)
dic5 = dict(zip(caqundic, np.arange(len(caqundic))))
d3['CAQUN'].replace(dic5, inplace=True)


# In[320]:


#cucy
d3['CUCY'] = d3['CUCY'].astype('str')
cucydic = np.load('cucydic.npy', allow_pickle=True)
dic6 = dict(zip(cucydic, np.arange(len(cucydic))))
d3['CUCY'].replace(dic6, inplace=True)


# In[321]:


#cntrno
d3['CNTRNO'][d3['CNTRNO'].isna()] = 0
d3['CNTRNO'][d3['CNTRNO'] != 0] = 1
d3['CNTRNO'] = d3['CNTRNO'].astype('str')


# In[322]:


#orgi
d3['ORGI'] = d3['ORGI'].astype('str')
d3['ORGI'][d3['ORGI'] == 'nan'] = ''
orgidic = np.load('orgidic.npy', allow_pickle=True)
dic7 = dict(zip(orgidic, np.arange(len(orgidic))))
d3['ORGI'].replace(dic7, inplace=True)


# In[323]:


#wwwmk
d3['WWWMK'] = d3['WWWMK'].astype('str')
d3['WWWMK'][d3['WWWMK'] == 'nan'] = ''
wwwdic = np.load('wwwdic.npy', allow_pickle=True)
dic8 = dict(zip(wwwdic, np.arange(len(wwwdic))))
d3['WWWMK'].replace(dic8, inplace=True)


# In[324]:


ordvnddic = np.load('ordvnddic.npy', allow_pickle=True)
d3['ORDVND'][~d3['ORDVND'].isin(ordvnddic)] = ''
dic9 = dict(zip(ordvnddic, np.arange(len(ordvnddic))))
d3['ORDVND'].replace(dic9, inplace=True)


# In[325]:


d3['COTYPE'][d3['COTYPE'].isna()] = 'A'
#COTYPEA 表示COTYPE是否為空值
d3['COTYPEA'] = d3['COTYPE'] == 'A'
#建立各供應商類別獨立欄位
d3['COTYPE1'] = d3['COTYPE'].str.contains('1')
d3['COTYPE2'] = d3['COTYPE'].str.contains('2')
d3['COTYPE3'] = d3['COTYPE'].str.contains('3')
d3['COTYPE4'] = d3['COTYPE'].str.contains('4')
d3['COTYPE5'] = d3['COTYPE'].str.contains('5')
d3 = d3.drop('COTYPE', axis=1)


# In[326]:


print('EDA finish')


# In[327]:


d3.head(100)


# In[328]:


#Load DNN model
model = k.models.load_model('dnn.h5')


# In[329]:


#Separate anser column and information colunm
d3X = d3.drop(['CPRUP', 'ORDCAQQTY', 'DLSTK', 'PURPRW','MKDYS', 'ANS', 'BG'], axis=1)
d3y = d3['ANS']


# In[330]:


d3X


# In[331]:


#Test Accuracy
model.evaluate(d3X, d3y)


# In[332]:


#Check confusion matrix
from sklearn.metrics import confusion_matrix
#Sensitivity can be altered by thereshold
classify = model.predict(d3X) > .08
classify = classify.astype('int')
confusion_matrix(d3y, classify)


# In[333]:


d3y.value_counts()

