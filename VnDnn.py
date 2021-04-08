
# coding: utf-8

# In[138]:


#載入packages
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, SimpleRNN, RNN
from keras.models import Model
import keras as k
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[93]:


#載入SQL擷取csv資料(檔案位置D://python/Vn2(all).csv)，並預覽
data = pd.read_csv('D://python/Vn(019-021).csv', encoding='ANSI')
data.head(12)


# In[94]:


#選擇數據分析有意義資料
#欄位對照：'PURID'採購別, 'PURPRW'採購人員, 'VNDNO'供應商, 'QLMK'品質異常註記, 'OVDMK'逾期註記, 'ANS'異常總註記(品質+交期), 'DLSTK'交貨庫別, 'SPECID'材料類別, 'ORDCAQQTY'訂購量,'CAQUN'訂購單位, 'CPRUP'材料單價, 'CUCY'訂購幣別, 'MKDYS'交貨天數, 'CNTRNO'合約編號, 'BRND'廠牌, 'ORGI'產地, 'WWWMK'網路註記
data2 = data[['PURID', 'PURPRW', 'VNDNO', 'QLMK', 'OVDMK', 'ANS', 'DLSTK', 'SPECID', 'ORDCAQQTY','CAQUN', 'CPRUP', 'CUCY', 'MKDYS', 'CNTRNO', 'BRND', 'ORGI', 'WWWMK', 'COTYPE', 'EMGPURMK', 'BGUP', 'BGCUCY']]


# In[95]:


#使用LabelEncoder預處理問字內容成分類類別
#採購別
le = LabelEncoder()
data2['PURID'] = le.fit_transform(data2['PURID'])
#儲存採購別對照供後續預測用
puriddic = le.classes_
np.save('puriddic', puriddic)
#採購人員
data2['PURPRW'] = data2['PURPRW'].astype('str')
data2['PURPRW'] = le.fit_transform(data2['PURPRW'])
#儲存採購人員對照供後續預測用
purprwdic = le.classes_
np.save('purprwdic', purprwdic)
#交貨庫
data2['DLSTK'] = le.fit_transform(data2['DLSTK'])
#儲存交貨庫對照供後續預測用
dlstkdic = le.classes_
np.save('dlstkdic', dlstkdic)
#材料類別
data2['SPECID'] = data2['SPECID'].str.slice(stop=2)
data2['SPECID'] = le.fit_transform(data2['SPECID'])
#儲存材料類別對照供後續預測用
speciddic = le.classes_
np.save('speciddic', speciddic)


# In[96]:


#供應商
#供應商類別過多，對交易次數<100供應商歸類為同一類
vndSelect = data2['VNDNO'].value_counts()
vndSelect = vndSelect[vndSelect.values > 100]
#迴圈做次數 <100區分
_ = []
for i in data2['VNDNO']:
    if i in vndSelect:
        _.append(i)
    else:
        _.append("")
#重新命名供應商為'ORDVND'
data2["ORDVND"] = _
data2["ORDVND"] = le.fit_transform(data2['ORDVND'])
#去除未轉換欄位
data2 = data2.drop('VNDNO', axis=1)
#儲存材料類別對照供後續預測用
ordvnddic = le.classes_
np.save('ordvnddic', ordvnddic)


# In[97]:


#預算
#預算差異量
data2["BG"] = data2['BGUP'][data2['CUCY'] == data2['BGCUCY']] - data2['CPRUP'][data2['CUCY'] == data2['BGCUCY']]


# In[98]:


#單位
data2['CAQUN'] = le.fit_transform(data2['CAQUN'])
#儲存單位類別對照供後續預測用
caqundic = le.classes_
np.save('caqundic', caqundic)
#幣別
data2['CUCY'] = le.fit_transform(data2['CUCY'])
#儲存幣別類別對照供後續預測用
cucydic = le.classes_
np.save('cucydic', cucydic)


# In[99]:


#以True/False區分有無合約
data2['CNTRNO'][data2['CNTRNO'].isna()] = 0
data2['CNTRNO'][data2['CNTRNO'] != 0] = 1


# In[100]:


#廠牌過於雜亂，捨棄不用
data2 = data2.drop('BRND', axis=1)


# In[101]:


#轉換為文字並去除'nan'浮點數
data2['ORGI'] = data2['ORGI'].astype('str')
data['ORGI'][data2['ORGI'] == 'nan'] = ''
#產地
data2['ORGI'] = le.fit_transform(data['ORGI'])
#儲存產地類別對照供後續預測用
orgidic = le.classes_
np.save('orgidic', orgidic)


# In[102]:


#網路註記去除浮點樹同上
data2['WWWMK'][data2['WWWMK'].isnull()] = ''


# In[103]:


#網路註記
data2['WWWMK'] = le.fit_transform(data2['WWWMK'])
#儲存網路註記類別對照供後續預測用
wwwdic = le.classes_
np.save('wwwdic', wwwdic)


# In[104]:


#供應商類型
data2["COTYPE"].value_counts()
# 1:製造商,2:經銷商,3:代理商,4:貿易商,5:其他
# Nan先處理為A類
data2['COTYPE'][data2['COTYPE'].isna()] = 'A'
#COTYPEA 表示COTYPE是否為空值
data2['COTYPEA'] = data2['COTYPE'] == 'A'
#建立各供應商類別獨立欄位
data2['COTYPE1'] = data2['COTYPE'].str.contains('1')
data2['COTYPE2'] = data2['COTYPE'].str.contains('2')
data2['COTYPE3'] = data2['COTYPE'].str.contains('3')
data2['COTYPE4'] = data2['COTYPE'].str.contains('4')
data2['COTYPE5'] = data2['COTYPE'].str.contains('5')
data2 = data2.drop('COTYPE', axis=1)


# In[105]:


#EMGPURMK 緊急註記預處理
data2['EMGPURMK'][data2['EMGPURMK']!='*'] = '0'
data2['EMGPURMK'][data2['EMGPURMK']=='*'] = '1'


# In[106]:


data2.head(50)


# In[107]:


#先取'CAQUN', 'CPRUP', 'MKDYS', 'ANS'以ML進行第1流程分析
from xgboost import XGBClassifier
clf = XGBClassifier()


# In[108]:


#data3為待分析數值DataFrame
data3 = data2[['CAQUN', 'CPRUP', 'MKDYS', 'BG', 'ANS']]


# In[109]:


#建立X:data3X；y:data3y
data3X = data3.drop('ANS', axis=1)
data3y = data3['ANS']


# In[110]:


#直接去訓練
clf.fit(data3X, data3y)


# In[111]:


#建立ans2為預測結果
ans2 = clf.predict(data3X)


# In[112]:


#加回原資料上，做為一項條件
data2['ANS2'] = ans2


# In[113]:


#轉換'PURID', 'PURPRW', 'DLSTK', 'SPECID', 'CAQUN', 'CUCY', 'CNTRNO', 'ORGI', 'WWWMK', 'ORDVND', 'ANS2'為文字類別(經過LabelEncoder為int)
data2[['PURID', 'PURPRW', 'DLSTK', 'SPECID', 'CAQUN', 'CUCY', 'CNTRNO', 'ORGI', 'WWWMK', 'ORDVND', 'ANS2']] = data2[['PURID', 'PURPRW', 'DLSTK', 'SPECID', 'CAQUN', 'CUCY', 'CNTRNO', 'ORGI', 'WWWMK', 'ORDVND', 'ANS2']].astype('str')


# In[114]:


#去除答案、已用於分析資料
data4 = data2.drop(['QLMK', 'OVDMK', 'ORDCAQQTY', 'MKDYS', 'CPRUP', 'DLSTK', 'PURPRW', 'BG', 'BGUP', 'BGCUCY'], axis=1)


# In[115]:


#用於DNN資料格式
print('Data ')
data4.head()


# In[116]:


#拆分訓練、測試資料
data4Xtrain, data4Xtest, data4ytrain, data4ytest = train_test_split(data4.drop('ANS', axis=1), data4['ANS'])


# In[117]:


data4Xtrain


# In[175]:


#Model building
inputlayer = Input(shape=(data4Xtrain.shape[-1], ))
dense1 = Dense(units=32, activation='sigmoid')(inputlayer)
dense2 = Dense(units=256, activation='sigmoid')(dense1)
dense3 = Dense(units=128, activation='sigmoid')(dense2)
dense5 = Dense(units=64, activation='sigmoid')(dense3)
dense6 = Dense(units=8, activation='sigmoid')(dense5)
output = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs=inputlayer, outputs = output)
print(model.summary())

model.compile(optimizer='RMSprop', loss='mse', metrics=['accuracy'])


# In[176]:


#訓練資料(每次輸入訓練800比，訓練500次)
history = model.fit(data4Xtrain, data4ytrain, verbose=1, batch_size=800, epochs=500, shuffle=True, workers=2)


# In[177]:


#測試資料準確率驗證
model.evaluate(data4Xtest, data4ytest, verbose=1)


# In[178]:


#以準確率/Epochs作圖
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=[30, 10])
plt.title('Accuracy', fontsize=50)
plt.xticks( fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.yticks( fontsize=20)
plt.ylabel('Ratio', fontsize=20)
plt.plot(history.history['accuracy'])

plt.show()


# In[179]:


#儲存XGBoost訓練參數
clf.save_model('xgboost1.h5')


# In[180]:


#儲存DNN訓練參數
model.save('dnn.h5')


# In[181]:


#預測資料機率已0.5為域值
score = model.predict(data4Xtest)
classcify = score > 0.2
#True/False轉換為1/0
classcify = classcify.astype('int')


# In[182]:


#已confusion matrix驗證準確率、召回率
from sklearn.metrics import confusion_matrix
confusion_matrix(data4ytest, classcify)

