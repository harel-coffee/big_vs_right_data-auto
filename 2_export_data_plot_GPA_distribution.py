import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

colors = '#2B8EE3', '#F9B43C', '#FA672F', '#BE1E2D', '#A7A9AC','#2B8EE3', '#F9B43C', '#FA672F', '#BE1E2D', '#A7A9AC'

plt.style.use('ggplot')
colors = [d['color'] for d in list(plt.rcParams['axes.prop_cycle'])]

#plt.rcParams['font.family'] = ''
#font_path = 'font/%s.ttf'
#prop = fm.FontProperties(fname=font_path % 'micross')
#mpl.rcParams['font.family'] = prop.get_name()


# # Load model output

# In[10]:


feat_sets = ['pre_all', 'post_all', 'combined_all', 'pre_elem', 'pre_elem_and_elem', 'task', 'non_task']

def load_output(m):    
    dfs = []
    for feat_set in feat_sets:
        for split_min in list(range(0,1000,200)):
            split_max = split_min+200
            
            filepath = 'model_output/model_%i_compact_%s_%i-%i.pkl' % (m, feat_set, split_min, split_max)
            if os.path.exists(filepath):
                dfs.append(pd.read_pickle(filepath))
    
    return pd.concat(dfs)

def load_output_bins(bins):    
    filepath = 'model_output/model_0_%ieven_bins_compact.pkl' % bins
    return pd.read_pickle(filepath).assign(bins=str(bins))

data = {
    'models': pd.concat(list(map(load_output, range(5))), sort=False).set_index(['clf','fs','poly_degree']).reset_index(),
    'bins': pd.concat(list(map(lambda b: load_output_bins(bins=b), range(3,6))), sort=False)\
             .append(load_output(0).assign(bins='3 - uneven'), sort=False)
}}

# # Structure model output

# In[11]:


hyperparams_export = ('clf__C',  'fs__max_features', 'clf__min_samples_split', 'clf__min_samples_leaf', 'clf__max_depth')
for label in ['models', 'bins']:
    for h in hyperparams_export: 
        data[label]['hyperparam_'+h] = data[label].param_opt.str[h]
    data[label]['hyperparam_clf__C'] = data[label]['hyperparam_clf__C'].round(7)
    data[label]['accuracy'] = data[label]['accuracy'].round(7)
    data[label] = data[label].drop(['feature_set','y_pred','pca','rfe','param_opt'],1).reset_index(drop=True)


# # Output structure model output

# In[12]:

    

data_split= {}
data_split['models_a'] = data['models'].iloc[:40000]
data_split['models_b'] = data['models'].iloc[40000:]
data_split['bins'] = data['bins']


for label in 'models_a', 'models_b', 'bins':
    s = s_explain+data_split[label].to_csv(index=False)
    with open(f'data/{label}.txt', 'w') as f:
        f.write(s)


# # Plot GPA distribution

# In[13]:


df = pd.read_pickle('data/preproc_compact.pkl')

f,ax = plt.subplots()


interval = sorted(pd.qcut(df.spring_2015_cum_gpa, q=[0,.2,.8,1]).unique())[1]


n_bins=12
hist, bin_edges = np.histogram(df.spring_2015_cum_gpa,bins=n_bins)
hist = hist/hist.sum()
hist = np.append(hist,0)
bin_edges = np.append(bin_edges,bin_edges[-1]+.1)
bins = [(bin_edges[i]+bin_edges[i+1])/2. for i in range(len(hist))]
xp = np.linspace(bin_edges[0],bin_edges[-1],300)
xp1 = np.linspace(bin_edges[0],interval.left,300)
xp2 = np.linspace(interval.left,interval.right,300)
xp3 = np.linspace(interval.right,bin_edges[-1],300)
pfit = np.poly1d(np.polyfit(bins,hist,5))
interval.left

plt.fill_between(xp1,pfit(xp1),alpha=1.,label="Low GPA")
plt.fill_between(xp2,pfit(xp2),alpha=1.,label="Medium GPA")
plt.fill_between(xp3,pfit(xp3),alpha=1.,label="High GPA")

linecolor = 'gray'
plt.plot([interval.left,interval.left],[0,pfit(interval.left)],"--",c=linecolor,linewidth=2)
plt.plot([interval.right,interval.right],[0,pfit(interval.right)],"--",c=linecolor,linewidth=2)
plt.plot(xp,pfit(xp),linewidth=2,c=linecolor)

#plt.hist(target_df[target_name],bins=n_bins)

plt.ylim(0,hist.max()*1.1)
plt.xlabel("Cumulative college GPA")
plt.ylabel("Density")
#plt.title("Distribution of GPAs")

plt.yticks([])
plt.legend()
f.savefig('output/distribution_GPA_color.pdf')

