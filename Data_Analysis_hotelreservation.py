#!/usr/bin/env python
# coding: utf-8

# # Hotel Reservation Classification Model

# ### Importing the libraries and Dataset

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


ds=pd.read_csv("D:\Mayank\Data science\Hotel Reservations.csv")
dscopy=ds.copy(deep=True)


# ### Reading the Data

# In[3]:


print(dscopy.head())


# In[4]:


print(dscopy.info())


# In[5]:


print(dscopy.describe())


# ### EDA

# In[6]:


dscopy.drop('Booking_ID',axis=1,inplace=True)


# In[7]:


for x in dscopy.columns:
    if(dscopy[(x)].dtype==object):
        print(x,'=',dscopy[(x)].unique())


# In[8]:


num=[]
cat=[]


# In[9]:


for m in dscopy.columns:
    if(dscopy[(m)].dtype==object):
        cat.append(m)
    else:
        num.append(m)


# In[10]:


print(num)
print(cat)


# In[22]:


print(dscopy.head(20))


# ### Data Visualization

# In[11]:


G1=sns.displot(dscopy['no_of_adults'])


# In[36]:


plt.figure(figsize=(12, 8))

# Create a countplot for room type
sns.countplot(x='room_type_reserved', data=ds, order=ds['room_type_reserved'].value_counts().index, palette='Set3')

# Set plot labels and title
plt.xlabel('Reserved Room Type')
plt.ylabel('Number of Bookings')
plt.title('Most Frequent Booked Room Type')

# Show the plot
plt.show()


# In[12]:


G2=sns.pairplot(dscopy[num],diag_kind='hist')


# In[24]:


guests_by_month_year = ds.groupby(['arrival_year', 'arrival_month']).size().reset_index(name='guest_count')

# Pivot the DataFrame for heatmap
guests_pivot = guests_by_month_year.pivot('arrival_month', 'arrival_year', 'guest_count')

# Set the size of the plot
plt.figure(figsize=(12, 8))

# Create a heatmap
sns.heatmap(guests_pivot, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Number of Guests'})

# Set plot labels and title
plt.xlabel('Arrival Year')
plt.ylabel('Arrival Month')
plt.title('Number of Guests or Reservations by Arrival Month and Year')

# Show the plot
plt.show()


# In[31]:


revenue_by_month = ds.groupby('arrival_month')['avg_price_per_room'].mean().reset_index(name='avg_price')
revenue_by_month['revenue'] = revenue_by_month['avg_price'] * ds.groupby('arrival_month')['no_of_week_nights'].sum()

# Set the size of the plot
plt.figure(figsize=(12, 8))

# Create a bar plot
sns.barplot(x='arrival_month', y='revenue', data=revenue_by_month, palette='viridis')

# Set plot labels and title
plt.xlabel('Arrival Month')
plt.ylabel('Revenue')
plt.title('Revenue by Arrival Month')

# Show the plot
plt.show()


# ### Training the model

# In[13]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in dscopy[cat]:
    dscopy[i]=le.fit_transform(dscopy[i])


# In[14]:


dscopy.head()


# In[15]:


x=dscopy.drop("booking_status",axis=1)
y=dscopy["booking_status"]


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
R=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
R.fit(x_train,y_train)
pred=R.predict(x_test)


# In[18]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,pred)
print(accuracy)


# ##### StandarScaler and pipeline increased the accuracy, so in future we can improve the accuracy even more with proper standarization and mean variance
