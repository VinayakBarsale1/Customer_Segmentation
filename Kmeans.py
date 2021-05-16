#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[71]:


data=pd.read_csv("Mall_Customers.csv")
data.head()


# In[72]:


data.shape
data.describe()


# In[73]:


data.dtypes


# In[74]:


data.isnull().sum(axis=0)


# In[75]:


data=data.drop(['CustomerID'],axis=1)


# In[76]:


data.head()


# In[ ]:





# In[77]:


data.head()


# In[79]:


import seaborn as sns

for i in range(1,4):
    sns.displot(data.iloc[:, i],kde=True)


# In[98]:


plt.figure()
sns.countplot(data['Gender'])
plt.show()


# In[96]:


plt.figure(1,figsize=(35,15))
n=1
for i in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    plt.subplot(1,3,n)
    plt.subplots_adjust(0.5,0.5)
    sns.violinplot(x=i,y='Gender',data=data)
    n=n+1
    
plt.show()


# In[99]:


corr=data.corr()
plt.figure()
sns.heatmap(corr)


# In[104]:


ai0_30=data['Annual Income (k$)'][(data['Annual Income (k$)']>=0 ) & (data['Annual Income (k$)']<30)]
ai30_60=data['Annual Income (k$)'][(data['Annual Income (k$)']>=30 ) & (data['Annual Income (k$)']<60)]
ai60_90=data['Annual Income (k$)'][(data['Annual Income (k$)']>=60 ) & (data['Annual Income (k$)']<90)]
ai90_120=data['Annual Income (k$)'][(data['Annual Income (k$)']>=90 ) & (data['Annual Income (k$)']<120)]
ai120_150=data['Annual Income (k$)'][(data['Annual Income (k$)']>=120 ) & (data['Annual Income (k$)']<150)]

axis_x=['ai0_30','ai30_60','ai60_90','ai90_120','ai120_150']
axis_y=[len(ai0_30.values),len(ai30_60.values),len(ai60_90.values),len(ai90_120.values),len(ai120_150.values)]

plt.figure()
sns.barplot(axis_x,axis_y)
plt.show()


# In[103]:


age_18_25=data['Age'][(data['Age']>=18 ) & (data['Age']<=25)]
age_26_35=data['Age'][(data['Age']>=26 ) & (data['Age']<=35)]
age_36_45=data['Age'][(data['Age']>=36 ) & (data['Age']<=45)]
age_46_55=data['Age'][(data['Age']>=46 ) & (data['Age']<=55)]
age_55_above=data['Age'][(data['Age']>55 )]

axis_x=['18-25','26-35','36-45','46-55','55>']
axis_y=[len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55_above.values)]

plt.figure()
sns.barplot(axis_x,axis_y)
plt.show()


# In[105]:


ss_0_20=data['Spending Score (1-100)'][(data['Spending Score (1-100)']>=0 ) & (data['Spending Score (1-100)']<20)]
ss_20_40=data['Spending Score (1-100)'][(data['Spending Score (1-100)']>=20 ) & (data['Spending Score (1-100)']<40)]
ss_40_60=data['Spending Score (1-100)'][(data['Spending Score (1-100)']>=40 ) & (data['Spending Score (1-100)']<60)]
ss_60_80=data['Spending Score (1-100)'][(data['Spending Score (1-100)']>=60 ) & (data['Spending Score (1-100)']<80)]
ss_80_100=data['Spending Score (1-100)'][(data['Spending Score (1-100)']>=80 ) & (data['Spending Score (1-100)']<100)]

axis_x=['0-20','20-40','40-60','60-80','80-100']
axis_y=[len(ss_0_20.values),len(ss_20_40.values),len(ss_40_60.values),len(ss_60_80.values),len(ss_80_100.values)]

plt.figure()
sns.barplot(axis_x,axis_y)
plt.show()


# In[107]:


data.head()
fdata=data.drop(['Gender'],axis=1)


# In[108]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler_Data=scaler.fit_transform(fdata)


# In[109]:


ss_data=pd.DataFrame(scaler_Data)


# In[115]:


alb=list()
from sklearn.cluster import KMeans

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=1234)
    kmeans.fit(ss_data)
    alb.append(kmeans.inertia_)
    
plt.plot(alb,marker='*',color='red')


# In[116]:


kmeans=KMeans(n_clusters=4,random_state=1234)
kmeans.fit(ss_data)


# In[118]:


labels=kmeans.labels_
labels=pd.DataFrame(labels)


# In[122]:


result_df=pd.concat([data,labels],axis=1)
result_df=result_df.rename(columns={0:'Labels'})


# In[134]:


from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)

ax.scatter(result_df['Age'],result_df['Spending Score (1-100)'],result_df['Annual Income (k$)'],c='red',marker='o')


# In[135]:


result_df.plot.scatter(x='Spending Score (1-100)',y='Annual Income (k$)',c='Labels',colormap='Accent')


# In[123]:


result_df.head()


# In[136]:


result_df.to_csv("Mall_Customers_data_with_Labels.csv")


# In[ ]:




