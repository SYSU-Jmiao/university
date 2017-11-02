
# coding: utf-8

# In[1]:


import os
import glob
glob.glob('./[0-9].*')

directories = glob.glob('../cwt_i_over_q/data/*')
print directories


# In[2]:


import numpy as np
files = np.asarray(list(map(lambda x:  glob.glob(x+'/*'),directories)))
print files.shape


# In[3]:


import random
map(lambda x:random.shuffle(x),files)


# In[35]:


train_files = np.asarray(list(map(lambda x:x[:10000], files)))
validation_files = np.asarray(list(map(lambda x:x[10000:], files)))


# In[36]:


def get_folder_name(prefix, file_path):
  name = os.path.basename(os.path.normpath(file_path))
  return os.path.join(prefix, name.split('_')[1])

def create_folders(prefix, files_list):
  folders = list(map(lambda x: get_folder_name(prefix, x[1]), files_list))
  for folder_name in folders:
     if not os.path.isdir(folder_name):
        os.makedirs(folder_name)


# In[41]:


def link_files(prefix, files_list):
    files = files_list[:,:].flatten();
    for f in files:
        folder = get_folder_name(prefix, f)
        name = os.path.basename(os.path.normpath(f))
        link = os.path.join(folder,name)
        if not os.path.islink(link):
            os.symlink(f,link)


# In[42]:


def prep_data_set(prefix, files):
    create_folders(prefix, files)
    link_files(prefix, files)


# In[43]:


prep_data_set('./data/train', train_files)


# In[44]:


prep_data_set('./data/validation', validation_files)


# In[ ]:
