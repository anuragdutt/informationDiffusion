#!/usr/bin/env python


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('../data/icpsr/DS0001/37070-0001-Data.tsv', sep='\t')

# In[3]:


# clean data
#df = df[(df['SCHRB']) & (df['SCHIDW2'] != 999)]

df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
df = df[df['SCHRB'].notna()]
df = df[(df['SCHIDW2'] != -98)]
df = df[(df['ID'] != 999)]

# In[4]:
print(df.head())
print(df.columns)
exit(0)

# figure out who the seeds were
seeds = df[(df['TREAT'] == 1) & (df['SCHTREAT'] == 1)]


# In[5]:


# each student gets a unique id
totalid = ((df['SCHIDW2'] * 1000) + pd.to_numeric(df['ID'], errors='coerce'))
# NAs get 99999
# but there are no missing values at this point (let alone exactly 7)
#totalid[totalid.isnull()] = np.arange(99999,99999+8)
# correction for 1242
totalid[totalid.index[totalid == 1242][1]] = 1242.2


#xyz = totalid['SCHIDW2']
#print(totalid[xyz.isin(xyz[xyz.duplicated()])].sort_values('SCHIDW2'))
#print(totalid[totalid.duplicated(keep=False)].sort_values())
#assert(False)

# In[ ]:


# Create matrix of student's nominations ("spend time with")
adjacency = df[['ST1',  'ST2',  'ST3',  'ST4',  'ST5',  'ST6',  'ST7',  'ST8',  'ST9', 'ST10']]
#network[pd.to_numeric(network['ST1'], errors='coerce').isnull()]['ST1']
adjacency = adjacency.apply(pd.to_numeric, errors='coerce')
# eliminate invalid (but numeric) responses
# supplementary analysis didn't do this and I'm not sure why
adjacency[adjacency < 0] = np.NaN
# assign unique schid to nominated students
#adjacency = adjacency + (df['SCHIDW2'] * 1000)

school_offset = df['SCHIDW2'] * 1000
for i in range(1,10+1):
    adjacency['ST'+str(i)] = adjacency['ST'+str(i)] + school_offset
#adjacency = adjacency.apply(lambda x : x + school_offset, axis='columns')

#print(adjacency.add(df['SCHIDW2'] * 1000, axis='columns').head())

# add student ID as index column
adjacency.insert(0, 'ID', totalid)

#print(adjacency.head())



#xyz = adjacency['ID']
#print(adjacency[xyz.isin(xyz[xyz.duplicated()])].sort_values('ID'))

adjacency = pd.wide_to_long(adjacency, ['ST'], i='ID', j='nth')

adjacency = adjacency.rename(columns = {'ST':'PEERID'})

adjacency.index = adjacency.index.droplevel(1)

adjacency['ID'] = adjacency.index

#print(adjacency[adjacency['ID']].head())

# more fixing for 1242
# this is taken from the original analysis/cleaning
# assumes that all nominations were for the student we
# reID'd as 1242.2 rather than the student we left
# as ID=1242, not sure where this assumption comes from
adjacency[adjacency['PEERID'] == 1242] = 1242.2

adjacency = adjacency.dropna()

adjacency['ID'] = adjacency['ID'].astype(str)
adjacency['PEERID'] = adjacency['PEERID'].astype(str)

adjacency.to_csv('../data/icpsr/DS0001/paluck-edgelist-seed.csv', index=False)





