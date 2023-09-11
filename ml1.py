#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sb


# In[8]:


import sklearn


# In[11]:


from sklearn import tree


# In[12]:


features = [[140,1],[130,1],[150,0],[170,0]]


# In[13]:


labels=[0,0,1,1]


# In[14]:


cif = tree.DecisionTreeClassifier()


# In[15]:


cif=cif.fit(features,labels)


# In[17]:


print(cif.predict([[150,0]]))


# In[ ]:




