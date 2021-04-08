
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import seaborn as sns
import re


# In[2]:


data = pd.read_csv("D://python/TW1.csv", encoding="ANSI")


# In[3]:


pd.set_option('display.max_rows', None)


# In[4]:


basicCount = data.count()["RCVNO"]
qlmkCount = data["QLMK"][data["QLMK"] == "*"].count()
ovdmkCount = data["OVDMK"][data["OVDMK"] == "*"].count()
qlmkr = qlmkCount/basicCount
ovdmkr = ovdmkCount/basicCount


# In[5]:


plt.figure(figsize=[8, 5])

plt.subplot(121)
plt.ylim(0, 15000)
plt.title("Abnormal counts")
plt.bar(["QLMK", "OVDMK"], [qlmkCount, ovdmkCount], width=[0.2, 0.2], color=['y', 'b'])

plt.subplot(122)
plt.ylim(0, .1)
plt.title("Abnormal ratio")
plt.bar(["QLMK", "OVDMK"], [qlmkr, ovdmkr], width=[0.2, 0.2], color=['y', 'b'])

plt.show()


# In[6]:


print("overall abnormal ratio")
print(data["ANS"].value_counts()[1]/basicCount)


# In[7]:


print("count of purprw")
data["PURPRW"].value_counts()


# In[8]:


combine1 = data[data["ANS"] == 1]["PURPRW"]
combine0 = data[data["ANS"] == 0]["PURPRW"]


# In[9]:


type(combine1)


# In[10]:


combine_D1 = pd.DataFrame({"PURPRW":combine1.value_counts().index, "COUNTS":combine1.value_counts().values})
combine_D0 = pd.DataFrame({"PURPRW":combine0.value_counts().index, "COUNTS":combine0.value_counts().values})


# In[11]:


combine1.value_counts().head()


# In[12]:


combine1.value_counts().index


# In[13]:


combine = pd.merge(combine_D0, combine_D1, how='left', on="PURPRW")


# In[14]:


plt.figure(figsize=[50, 15], edgecolor='white')
plt.title("Abnormal purno counts",fontsize=50)
plt.xticks(fontsize=20, rotation=90)
plt.bar(combine["PURPRW"], combine['COUNTS_x'])
plt.bar(combine["PURPRW"], combine['COUNTS_y'], bottom=combine['COUNTS_x'])
plt.show()


# In[15]:


ratio = combine["COUNTS_y"]/combine["COUNTS_x"]


# In[16]:


plt.figure(figsize=[50, 15], edgecolor='white')
plt.title("Abnormal purno rate",fontsize=50)
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.bar(combine["PURPRW"], ratio,color='gold')
plt.show()


# In[17]:


combine500 = combine[combine["COUNTS_x"] >= 500]


# In[18]:


plt.figure(figsize=[50, 15], edgecolor='white')
plt.title("Abnormal purno counts >500",fontsize=50)
plt.xticks(fontsize=30, rotation=90)
plt.bar(combine500["PURPRW"], combine500['COUNTS_x'])
plt.bar(combine500["PURPRW"], combine500['COUNTS_y'], bottom=combine500['COUNTS_x'])
plt.show()


# In[19]:


ratio500 = combine500["COUNTS_y"]/combine500["COUNTS_x"]


# In[20]:


plt.figure(figsize=[50, 15], edgecolor='white')
plt.title("Abnormal purno rate >500",fontsize=50)
plt.xticks(fontsize=20, rotation=90)
plt.yticks(fontsize=20)
plt.bar(combine500["PURPRW"], ratio500,color='gold')
plt.show()


# In[21]:


dataCo = data[["CO", "ANS"]]


# In[22]:


dataCo0 = dataCo[dataCo["ANS"]==0]
dataCo1 = dataCo[dataCo["ANS"]==1]
dataframeCo0 = pd.DataFrame({"CO":dataCo0["CO"].value_counts().index, 'COUNTS':dataCo0["CO"].value_counts().values})
dataframeCo1 = pd.DataFrame({"CO":dataCo1["CO"].value_counts().index, 'COUNTS':dataCo1["CO"].value_counts().values})


# In[23]:


dataframeCoM = pd.merge(dataframeCo0, dataframeCo1, how="left", on="CO", sort=True)


# In[24]:


dataframeCoM[dataframeCoM["CO"] == "0"]


# In[25]:


plt.figure(figsize=[50, 15])
plt.title("Abnormal counts by CO", fontsize=50)
plt.xticks(fontsize=30, rotation=90)
plt.yticks(fontsize=20)
plt.bar(dataframeCoM['CO'], dataframeCoM['COUNTS_x'])
plt.bar(dataframeCoM['CO'], dataframeCoM['COUNTS_y'], bottom=dataframeCoM['COUNTS_x'])
plt.show()


# In[26]:


dataDistinctCo_Y =pd.DataFrame({'COUNT':data[data["ANS"] == 1]["CO"].value_counts().values, "CO":data[data["ANS"] == 1]["CO"].value_counts().index})
dataDistinctCo_N =pd.DataFrame({'COUNT':data[data["ANS"] == 0]["CO"].value_counts().values, "CO":data[data["ANS"] == 0]["CO"].value_counts().index})


# In[27]:


dataDistinctCo = pd.merge(dataDistinctCo_N, dataDistinctCo_Y, how='left', on="CO").fillna(0)


# In[28]:


dataDistinctCo["RATE"] = dataDistinctCo["COUNT_y"]/(dataDistinctCo["COUNT_x"] + dataDistinctCo["COUNT_y"])


# In[29]:


dataDistinctCo = dataDistinctCo[dataDistinctCo["COUNT_x"] >= 500]


# In[30]:


plt.figure(figsize=[50, 15])
plt.title("Abnormal rate by CO(>500)", fontsize=50)
plt.xticks(fontsize=30, rotation=90)
plt.yticks(fontsize=20)
plt.bar(dataDistinctCo['CO'], dataDistinctCo['RATE'])
plt.show()


# In[31]:


#Plt of Vnd abnormal 
dataVnd = data[data["ORDVND"].isin(data["ORDVND"].value_counts()[data["ORDVND"].value_counts() >= 1000].index)]


# In[32]:


datVnd1 = dataVnd[data["ANS"] == 1]["ORDVND"].value_counts()
datVnd0 = dataVnd[data["ANS"] == 0]["ORDVND"].value_counts()


# In[33]:


datVnd1 = pd.DataFrame({"CO":datVnd1.index, "COUNTS":datVnd1.values})
datVnd0 = pd.DataFrame({"CO":datVnd0.index, "COUNTS":datVnd0.values})


# In[34]:


pltdtaVnd = pd.merge(datVnd0, datVnd1, how='left', on="CO").fillna(0)


# In[35]:


plt.figure(figsize=[50, 15])
plt.title("Abnormal counts by ORDVND", fontsize=50)
plt.xticks(fontsize=30, rotation=90)
plt.yticks(fontsize=30)
plt.bar(pltdtaVnd["CO"], pltdtaVnd["COUNTS_x"])
plt.bar(pltdtaVnd["CO"], pltdtaVnd["COUNTS_y"], bottom=pltdtaVnd["COUNTS_x"])
plt.show()


# In[36]:


pltdtaVnd["RATE"] = pltdtaVnd["COUNTS_y"] / (pltdtaVnd['COUNTS_x'] + pltdtaVnd['COUNTS_y'])


# In[37]:


plt.figure(figsize=[50, 15])
plt.title("Abnormal rate by ORDVND", fontsize=50)
plt.xticks(fontsize=30, rotation=90)
plt.yticks(fontsize=30)
plt.bar(pltdtaVnd["CO"], pltdtaVnd["RATE"], color='r')
plt.show()


# In[38]:


#Cntrno vs ANS
dataC = data[["CNTRNO", "ANS"]]
dataC.head()


# In[39]:


pltN = dataC[dataC["CNTRNO"].isnull()]["ANS"].value_counts()
pltC = dataC[dataC["CNTRNO"].notnull()]["ANS"].value_counts()
pltC.index = pltC.index.astype('int')


# In[40]:


plt.figure(figsize=[5, 10])
plt.title("Abnormal counts by CONTRNO", fontsize=30)
plt.xticks((0.5, 1), ('CNT', 'N CNT'))
_ = plt.bar([0.5,1], [pltC[0], pltN[0]], width=0.49)
_1 = plt.bar([0.5,1], [pltC[1], pltN[1]], width=0.49, bottom=[pltC[0], pltN[0]], color='r')
plt.legend((_, _1), ("Normal", 'Abnormal'))
plt.show()


# In[41]:


#Analysis APKID
dataA = data[["ANS", "APKID"]]
dataA.head()


# In[42]:


dataAB = dataA[dataA["APKID"].isnull()]
dataAF = dataA[dataA["APKID"].notnull()]


# In[43]:


dataAB["ANS"].value_counts()


# In[44]:


dataAF["ANS"].count()


# In[45]:


plt.figure(figsize=[12, 6])

plt.subplot(121)
plt.title("Counts with/without APKID")
_0 = plt.bar([0.75,1.75], [dataAB['ANS'].value_counts()[0], dataAF['ANS'].value_counts()[0]], width=0.25,color='b')
_1 = plt.bar([1, 2], [dataAB['ANS'].value_counts()[1],dataAB['ANS'].value_counts()[1]], width=0.25, color='orange')
plt.legend((_0, _1), ("Normal", "Abnormal"))
plt.xticks((0.875, 1.875), ("NO APKID", "WITH APKID"))

plt.subplot(122)
plt.title("Abnormal rate with/without APKID")
_3 = plt.bar([0.75, 1.75], [dataAB["ANS"].value_counts()[1]/dataAB["ANS"].count(), dataAF["ANS"].value_counts()[1]/dataAF["ANS"].count()])
plt.xticks((0.875, 1.875), ("NO APKID", "WITH APKID"))
plt.show()


# In[46]:


def apkid_splitter(n):#n is the series wanted to be split
    apk = n['APKID'].astype('str')
    list2 = []
    for i in data.index:
        if apk[i] == 'nan':
            list2 += ["NoApk"]
        else:
            _ = []
            for j in np.arange(0, len(apk[i]) , 2):
                _ += [apk[i][j:j+2]]
            list2 += [_]
    return list2
    print("Susscess")


# In[47]:


list2 = apkid_splitter(data)


# In[48]:


df = pd.DataFrame(list2)


# In[49]:


df3 = pd.concat([df, data['ANS']], axis=1)


# In[50]:


df4 = df3.explode(0, ignore_index=True)


# In[51]:


df4.head(100)


# In[52]:


for m in range(len(df4)):
    if len(df4[0][m]) == 1:
        df4[0][m] = df4[0][m] + " "
    else:
        continue


# In[53]:


group = df4.groupby(0)


# In[54]:


group.sum()


# In[55]:


len(group.sum()['ANS'].values / group.count()['ANS'].values)


# In[56]:


testrate = group.sum()['ANS'] / group.count()['ANS']


# In[57]:


plt.figure(figsize=[40, 10])
plt.subplot(211)
plt.title("APKIDs", fontsize=50)
plt.xticks((np.arange(1, len(group.count()+1))), (group.count().index), fontsize = 10)
nums = plt.bar(np.arange(0.75, len(group.count()) + 0.75, 1), group.count()['ANS'], width = 0.5, color='b')
eros = plt.bar(np.arange(1.25, len(group.sum()) + 1.25, 1), group.sum()['ANS'], width = 0.5, color='y')

plt.subplot(212)
plt.xticks((np.arange(1, len(group.count() +1))), (testrate.index), fontsize = 10)
rate = plt.bar(testrate.index, group.sum()['ANS'] / group.count()['ANS'], width=1)

plt.show()


# In[58]:


#PURID
dataPurid = data[['ANS', 'PURID']]


# In[59]:


len(dataPurid)


# In[60]:


dataPuridCounts = dataPurid['PURID'].value_counts()
dataPuridCounts


# In[61]:


type(dataPuridCounts)


# In[62]:


dataPuridG = dataPurid.groupby('PURID')


# In[63]:


plotID = dataPuridG.sum()
plotID['ANS']


# In[64]:


plotAll = plotID.join(dataPuridCounts, how='left')


# In[65]:


plotAll


# In[66]:


plotAll.index.astype('str')


# In[67]:


plt.figure(figsize=(30, 15))

plt.subplot(121)
plt.title("COUNTS", fontsize=30)
plt.xticks(np.arange(0.75, 5, 1), plotAll.index.astype('str'), fontsize=20)
total = plt.bar(np.arange(0.75, 5, 1), plotAll['PURID'].values, width=0.25)
error = plt.bar(np.arange(1, 6, 1), plotAll['ANS'].values, width=0.25)
plt.legend((total, error), ("PURID", "ERROR COUNTS"))

plt.subplot(122)
plt.title("RATIO", fontsize=30)
plt.xticks(np.arange(1, 6, 1), plotAll.index.astype('str'), fontsize=20)
plt.bar(range(1, 6, 1), plotAll['ANS'] / plotAll['PURID'])


plt.show()


# In[68]:


plotAll['PURID'].values


# In[69]:


plotAll['ANS'] / plotAll['PURID']

