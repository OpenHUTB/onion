#!/usr/bin/env python
# coding: utf-8

# In[43]:


import os 
import glob
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.graphics.factorplots import interaction_plot
from pingouin import mixed_anova
from pingouin import rm_anova
from pingouin import compute_effsize
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import scipy.stats as st


# In[44]:


acc_FRtoFR_N = []
target_group = 'FR'
lang_name = 'FR'
condition_name = 'N'
sub_exclude = []

for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_FRtoFR_N.append(accuracy)


# In[45]:


acc_FRtoJP_N = []
target_group = 'FR'
lang_name = 'JP'
condition_name = 'N'
sub_exclude = []

for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_FRtoJP_N.append(accuracy)


# In[46]:


acc_FRtoFR_J = []
target_group = 'FR'
lang_name = 'FR'
condition_name = 'J'
sub_exclude = []

for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_FRtoFR_J.append(accuracy)


# In[47]:


acc_FRtoJP_J = []
target_group = 'FR'
lang_name = 'JP'
condition_name = 'J'
sub_exclude = []

for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_FRtoJP_J.append(accuracy)


# In[48]:


acc_FRtoFR_S = []
target_group = 'FR'
lang_name = 'FR'
condition_name = 'S'
sub_exclude = []

for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_FRtoFR_S.append(accuracy)


# In[49]:


acc_FRtoJP_S = []
target_group = 'FR'
lang_name = 'JP'
condition_name = 'S'
sub_exclude = []

for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_FRtoJP_S.append(accuracy)


# In[50]:


acc_FRtoFR_R = []
target_group = 'FR'
lang_name = 'FR'
condition_name = 'R'
sub_exclude = []

for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_FRtoFR_R.append(accuracy)


# In[51]:


acc_FRtoJP_R = []
target_group = 'FR'
lang_name = 'JP'
condition_name = 'R'
sub_exclude = []

for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_FRtoJP_R.append(accuracy)


# In[ ]:





# In[52]:


acc_JPtoFR_N = []
target_group = 'JP'
lang_name = 'FR'
condition_name = 'N'
sub_exclude = [5]

for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_JPtoFR_N.append(accuracy)


# In[53]:


acc_JPtoJP_N = []
target_group = 'JP'
lang_name = 'JP'
condition_name = 'N'
sub_exclude = [5]

for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_JPtoJP_N.append(accuracy)


# In[54]:


acc_JPtoFR_J = []
target_group = 'JP'
lang_name = 'FR'
condition_name = 'J'
sub_exclude = [5]

for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_JPtoFR_J.append(accuracy)


# In[55]:


acc_JPtoJP_J = []
target_group = 'JP'
lang_name = 'JP'
condition_name = 'J'
sub_exclude = [5]

for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_JPtoJP_J.append(accuracy)


# In[56]:


acc_JPtoFR_S = []
target_group = 'JP'
lang_name = 'FR'
condition_name = 'S'
sub_exclude = [5]

for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_JPtoFR_S.append(accuracy)


# In[57]:


acc_JPtoJP_S = []
target_group = 'JP'
lang_name = 'JP'
condition_name = 'S'
sub_exclude = [5]

for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_JPtoJP_S.append(accuracy)


# In[58]:


acc_JPtoFR_R = []
target_group = 'JP'
lang_name = 'FR'
condition_name = 'R'
sub_exclude = [5]

for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_JPtoFR_R.append(accuracy)


# In[59]:


acc_JPtoJP_R = []
target_group = 'JP'
lang_name = 'JP'
condition_name = 'R'
sub_exclude = [5]

for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        stim_names = dat['soundfile']
        targ_list = []
        for ss in range(len(stim_names)):
            if stim_names[ss].split('_')[0] == lang_name:
                if stim_names[ss].split('_')[2][0] == condition_name:
                    targ_list.append(ss)    
        dat_targ = dat.iloc[targ_list]
        accuracy = np.mean(dat_targ['response'] == 'correct')*100   
        acc_JPtoJP_R.append(accuracy)


# In[ ]:





# In[60]:


mean_FRtoFR_N = np.mean(acc_FRtoFR_N)
ci_FRtoFR_N = st.t.interval(alpha=0.95, df=len(acc_FRtoFR_N)-1, loc=np.mean(acc_FRtoFR_N), scale=st.sem(acc_FRtoFR_N))
mean_FRtoJP_N = np.mean(acc_FRtoJP_N)
ci_FRtoJP_N = st.t.interval(alpha=0.95, df=len(acc_FRtoJP_N)-1, loc=np.mean(acc_FRtoJP_N), scale=st.sem(acc_FRtoJP_N))

mean_FRtoFR_J = np.mean(acc_FRtoFR_J)
ci_FRtoFR_J = st.t.interval(alpha=0.95, df=len(acc_FRtoFR_J)-1, loc=np.mean(acc_FRtoFR_J), scale=st.sem(acc_FRtoFR_J))
mean_FRtoJP_J = np.mean(acc_FRtoJP_J)
ci_FRtoJP_J = st.t.interval(alpha=0.95, df=len(acc_FRtoJP_J)-1, loc=np.mean(acc_FRtoJP_J), scale=st.sem(acc_FRtoJP_J))

mean_FRtoFR_S = np.mean(acc_FRtoFR_S)
ci_FRtoFR_S = st.t.interval(alpha=0.95, df=len(acc_FRtoFR_S)-1, loc=np.mean(acc_FRtoFR_S), scale=st.sem(acc_FRtoFR_S))
mean_FRtoJP_S = np.mean(acc_FRtoJP_S)
ci_FRtoJP_S = st.t.interval(alpha=0.95, df=len(acc_FRtoJP_S)-1, loc=np.mean(acc_FRtoJP_S), scale=st.sem(acc_FRtoJP_S))

mean_FRtoFR_R = np.mean(acc_FRtoFR_R)
ci_FRtoFR_R = st.t.interval(alpha=0.95, df=len(acc_FRtoFR_R)-1, loc=np.mean(acc_FRtoFR_R), scale=st.sem(acc_FRtoFR_R))
mean_FRtoJP_R = np.mean(acc_FRtoJP_R)
ci_FRtoJP_R = st.t.interval(alpha=0.95, df=len(acc_FRtoJP_R)-1, loc=np.mean(acc_FRtoJP_R), scale=st.sem(acc_FRtoJP_R))



# In[ ]:





# In[61]:


mean_JPtoFR_N = np.mean(acc_JPtoFR_N)
ci_JPtoFR_N = st.t.interval(alpha=0.95, df=len(acc_JPtoFR_N)-1, loc=np.mean(acc_JPtoFR_N), scale=st.sem(acc_JPtoFR_N))
mean_JPtoJP_N = np.mean(acc_JPtoJP_N)
ci_JPtoJP_N = st.t.interval(alpha=0.95, df=len(acc_JPtoJP_N)-1, loc=np.mean(acc_JPtoJP_N), scale=st.sem(acc_JPtoJP_N))

mean_JPtoFR_J = np.mean(acc_JPtoFR_J)
ci_JPtoFR_J = st.t.interval(alpha=0.95, df=len(acc_JPtoFR_J)-1, loc=np.mean(acc_JPtoFR_J), scale=st.sem(acc_JPtoFR_J))
mean_JPtoJP_J = np.mean(acc_JPtoJP_J)
ci_JPtoJP_J = st.t.interval(alpha=0.95, df=len(acc_JPtoJP_J)-1, loc=np.mean(acc_JPtoJP_J), scale=st.sem(acc_JPtoJP_J))

mean_JPtoFR_S = np.mean(acc_JPtoFR_S)
ci_JPtoFR_S = st.t.interval(alpha=0.95, df=len(acc_JPtoFR_S)-1, loc=np.mean(acc_JPtoFR_S), scale=st.sem(acc_JPtoFR_S))
mean_JPtoJP_S = np.mean(acc_JPtoJP_S)
ci_JPtoJP_S = st.t.interval(alpha=0.95, df=len(acc_JPtoJP_S)-1, loc=np.mean(acc_JPtoJP_S), scale=st.sem(acc_JPtoJP_S))

mean_JPtoFR_R = np.mean(acc_JPtoFR_R)
ci_JPtoFR_R = st.t.interval(alpha=0.95, df=len(acc_JPtoFR_R)-1, loc=np.mean(acc_JPtoFR_R), scale=st.sem(acc_JPtoFR_R))
mean_JPtoJP_R = np.mean(acc_JPtoJP_R)
ci_JPtoJP_R = st.t.interval(alpha=0.95, df=len(acc_JPtoJP_R)-1, loc=np.mean(acc_JPtoJP_R), scale=st.sem(acc_JPtoJP_R))


# In[ ]:





# In[62]:


xpos_FR = np.array([0, 0.7])
xpos_JP = np.array([0.3, 1])
xpos_label = np.array([0.3, 1])
width = 0.3
labels = ['FR', 'JP']

figure, ax = plt.subplots(figsize=[3,4])
ax.bar(xpos_FR, [mean_FRtoFR_N, mean_JPtoFR_N], color=[1,1,1], edgecolor ='black', width=width, align='center')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FRtoFR_N, color=[0.3,0.3,0.3])
ax.plot([xpos_FR[1],xpos_FR[1]], ci_JPtoFR_N, color=[0.3,0.3,0.3])

ax.bar(xpos_JP, [mean_FRtoJP_N, mean_JPtoJP_N], color=[0.5,0.5,0.5], edgecolor ='black', width=width, align='center')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_FRtoJP_N, color=[0.3,0.3,0.3])
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JPtoJP_N, color=[0.3,0.3,0.3])


plt.xlabel('Participant group', fontname='Arial')
plt.ylabel('Accuracy (%)', fontname='Arial')
plt.xticks(xpos_label, labels, fontname='Arial')
plt.yticks([0, 20, 40, 60, 80], fontname='Arial')
plt.tight_layout()

plt.show()
fname = 'Exp2_Barplot_Normal.pdf'
figure.savefig(fname, format='pdf')


# In[63]:


acc_all_N = np.concatenate([acc_FRtoFR_N, acc_FRtoJP_N, acc_JPtoJP_N, acc_JPtoFR_N])
#1...FR, 0...JP
subject_type = np.concatenate([np.ones(len(acc_FRtoFR_N) * 2), np.zeros(len(acc_JPtoJP_N) * 2)])
lang_type =  np.concatenate([np.ones(len(acc_FRtoFR_N)), np.zeros(len(acc_FRtoFR_N)), np.zeros(len(acc_JPtoJP_N)), np.ones(len(acc_JPtoJP_N))])
sub_id = np.concatenate([list(range(0,len(acc_FRtoFR_N))), list(range(0,len(acc_FRtoFR_N))),  list(range(len(acc_FRtoFR_N),len(acc_FRtoFR_N)+len(acc_JPtoJP_N))),  list(range(len(acc_FRtoFR_N),len(acc_FRtoFR_N)+len(acc_JPtoJP_N)))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': acc_all_N, 'subject_type': subject_type, 'lang_type': lang_type})

aov = mixed_anova(dv='hit_rate', between='subject_type', within='lang_type', subject='sub_id', data=df)
aov.round(5)


# In[64]:


t_score, p_value = ttest_rel(acc_FRtoFR_N, acc_FRtoJP_N, alternative = 'greater')
cohen_d = compute_effsize(acc_FRtoFR_N, acc_FRtoJP_N, paired=True, eftype='cohen')
print("FR sub => FR vs JP stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_JPtoJP_N, acc_JPtoFR_N, alternative = 'greater')
cohen_d = compute_effsize(acc_JPtoJP_N, acc_JPtoFR_N, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[ ]:





# In[65]:


xpos_FR = np.array([0, 0.7])
xpos_JP = np.array([0.3, 1])
xpos_label = np.array([0.15, 0.85])
width = 0.3
labels = ['FR', 'JP']

figure, ax = plt.subplots(figsize=[3,4])
ax.bar(xpos_FR, [mean_FRtoFR_J, mean_JPtoFR_J], color=[1,1,1], edgecolor ='black', width=width, align='center')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FRtoFR_J, color=[0.3,0.3,0.3])
ax.plot([xpos_FR[1],xpos_FR[1]], ci_JPtoFR_J, color=[0.3,0.3,0.3])

ax.bar(xpos_JP, [mean_FRtoJP_J, mean_JPtoJP_J], color=[0.5,0.5,0.5], edgecolor ='black', width=width, align='center')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_FRtoJP_J, color=[0.3,0.3,0.3])
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JPtoJP_J, color=[0.3,0.3,0.3])


plt.xlabel('Participant group', fontname='Arial')
plt.ylabel('Accuracy (%)', fontname='Arial')
plt.xticks(xpos_label, labels, fontname='Arial')
plt.yticks([0, 20, 40, 60, 80], fontname='Arial')
plt.tight_layout()

plt.show()
fname = 'Exp2_Barplot_Jabberwocky.pdf'
figure.savefig(fname, format='pdf')


# In[66]:


acc_all_J = np.concatenate([acc_FRtoFR_J, acc_FRtoJP_J, acc_JPtoJP_J, acc_JPtoFR_J])
#1...FR, 0...JP
subject_type = np.concatenate([np.ones(len(acc_FRtoFR_N) * 2), np.zeros(len(acc_JPtoJP_N) * 2)])
lang_type =  np.concatenate([np.ones(len(acc_FRtoFR_N)), np.zeros(len(acc_FRtoFR_N)), np.zeros(len(acc_JPtoJP_N)), np.ones(len(acc_JPtoJP_N))])
sub_id = np.concatenate([list(range(0,len(acc_FRtoFR_N))), list(range(0,len(acc_FRtoFR_N))),  list(range(len(acc_FRtoFR_N),len(acc_FRtoFR_N)+len(acc_JPtoJP_N))),  list(range(len(acc_FRtoFR_N),len(acc_FRtoFR_N)+len(acc_JPtoJP_N)))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': acc_all_J, 'subject_type': subject_type, 'lang_type': lang_type})

aov = mixed_anova(dv='hit_rate', between='subject_type', within='lang_type', subject='sub_id', data=df)
aov.round(5)


# In[67]:


t_score, p_value = ttest_rel(acc_FRtoFR_J, acc_FRtoJP_J, alternative = 'greater')
cohen_d = compute_effsize(acc_FRtoFR_J, acc_FRtoJP_J, paired=True, eftype='cohen')
print("FR sub => FR vs JP stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_JPtoJP_J, acc_JPtoFR_J, alternative = 'greater')
cohen_d = compute_effsize(acc_JPtoJP_J, acc_JPtoFR_J, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[68]:


xpos_FR = np.array([0, 0.7])
xpos_JP = np.array([0.3, 1])
xpos_label = np.array([0.15, 0.85])
width = 0.3
labels = ['FR', 'JP']

figure, ax = plt.subplots(figsize=[3,4])
ax.bar(xpos_FR, [mean_FRtoFR_S, mean_JPtoFR_S], color=[1,1,1], edgecolor ='black', width=width, align='center')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FRtoFR_S, color=[0.3,0.3,0.3])
ax.plot([xpos_FR[1],xpos_FR[1]], ci_JPtoFR_S, color=[0.3,0.3,0.3])

ax.bar(xpos_JP, [mean_FRtoJP_S, mean_JPtoJP_S], color=[0.5,0.5,0.5], edgecolor ='black', width=width, align='center')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_FRtoJP_S, color=[0.3,0.3,0.3])
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JPtoJP_S, color=[0.3,0.3,0.3])


plt.xlabel('Participant group', fontname='Arial')
plt.ylabel('Accuracy (%)', fontname='Arial')
plt.xticks(xpos_label, labels, fontname='Arial')
plt.yticks([0, 20, 40, 60, 80], fontname='Arial')
plt.tight_layout()

plt.show()
fname = 'Exp2_Barplot_Shuffled.pdf'
figure.savefig(fname, format='pdf')


# In[69]:


acc_all_S = np.concatenate([acc_FRtoFR_S, acc_FRtoJP_S, acc_JPtoJP_S, acc_JPtoFR_S])
#1...FR, 0...JP
subject_type = np.concatenate([np.ones(len(acc_FRtoFR_N) * 2), np.zeros(len(acc_JPtoJP_N) * 2)])
lang_type =  np.concatenate([np.ones(len(acc_FRtoFR_N)), np.zeros(len(acc_FRtoFR_N)), np.zeros(len(acc_JPtoJP_N)), np.ones(len(acc_JPtoJP_N))])
sub_id = np.concatenate([list(range(0,len(acc_FRtoFR_N))), list(range(0,len(acc_FRtoFR_N))),  list(range(len(acc_FRtoFR_N),len(acc_FRtoFR_N)+len(acc_JPtoJP_N))),  list(range(len(acc_FRtoFR_N),len(acc_FRtoFR_N)+len(acc_JPtoJP_N)))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': acc_all_S, 'subject_type': subject_type, 'lang_type': lang_type})

aov = mixed_anova(dv='hit_rate', between='subject_type', within='lang_type', subject='sub_id', data=df)
aov.round(5)


# In[70]:


t_score, p_value = ttest_rel(acc_FRtoFR_S, acc_FRtoJP_S, alternative = 'greater')
cohen_d = compute_effsize(acc_FRtoFR_S, acc_FRtoJP_S, paired=True, eftype='cohen')
print("FR sub => FR vs JP stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_JPtoJP_S, acc_JPtoFR_S, alternative = 'greater')
cohen_d = compute_effsize(acc_JPtoJP_S, acc_JPtoFR_S, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[71]:


xpos_FR = np.array([0, 0.7])
xpos_JP = np.array([0.3, 1])
xpos_label = np.array([0.15, 0.85])
width = 0.3
labels = ['FR', 'JP']

figure, ax = plt.subplots(figsize=[3,4])
ax.bar(xpos_FR, [mean_FRtoFR_R, mean_JPtoFR_R], color=[1,1,1], edgecolor ='black', width=width, align='center')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FRtoFR_R, color=[0.3,0.3,0.3])
ax.plot([xpos_FR[1],xpos_FR[1]], ci_JPtoFR_R, color=[0.3,0.3,0.3])

ax.bar(xpos_JP, [mean_FRtoJP_R, mean_JPtoJP_R], color=[0.5,0.5,0.5], edgecolor ='black', width=width, align='center')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_FRtoJP_R, color=[0.3,0.3,0.3])
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JPtoJP_R, color=[0.3,0.3,0.3])


plt.xlabel('Participant group', fontname='Arial')
plt.ylabel('Accuracy (%)', fontname='Arial')
plt.xticks(xpos_label, labels, fontname='Arial')
plt.yticks([0, 20, 40, 60, 80], fontname='Arial')
plt.tight_layout()

plt.show()
fname = 'Exp2_Barplot_Reversed.pdf'
figure.savefig(fname, format='pdf')


# In[72]:


acc_all_R = np.concatenate([acc_FRtoFR_R, acc_FRtoJP_R, acc_JPtoJP_R, acc_JPtoFR_R])
#1...FR, 0...JP
subject_type = np.concatenate([np.ones(len(acc_FRtoFR_N) * 2), np.zeros(len(acc_JPtoJP_N) * 2)])
lang_type =  np.concatenate([np.ones(len(acc_FRtoFR_N)), np.zeros(len(acc_FRtoFR_N)), np.zeros(len(acc_JPtoJP_N)), np.ones(len(acc_JPtoJP_N))])
sub_id = np.concatenate([list(range(0,len(acc_FRtoFR_N))), list(range(0,len(acc_FRtoFR_N))),  list(range(len(acc_FRtoFR_N),len(acc_FRtoFR_N)+len(acc_JPtoJP_N))),  list(range(len(acc_FRtoFR_N),len(acc_FRtoFR_N)+len(acc_JPtoJP_N)))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': acc_all_R, 'subject_type': subject_type, 'lang_type': lang_type})

aov = mixed_anova(dv='hit_rate', between='subject_type', within='lang_type', subject='sub_id', data=df)
aov.round(5)


# In[73]:


t_score, p_value = ttest_rel(acc_FRtoFR_R, acc_FRtoJP_R, alternative = 'greater')
cohen_d = compute_effsize(acc_FRtoFR_R, acc_FRtoJP_R, paired=True, eftype='cohen')
print("FR sub => FR vs JP stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_JPtoJP_R, acc_JPtoFR_R, alternative = 'greater')
cohen_d = compute_effsize(acc_JPtoJP_R, acc_JPtoFR_R, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[ ]:





# In[81]:


acc_all_FR = np.vstack([acc_FRtoFR_N, acc_FRtoJP_N, 
                       acc_FRtoFR_J, acc_FRtoJP_J,
                       acc_FRtoFR_S, acc_FRtoJP_S,
                       acc_FRtoFR_R, acc_FRtoJP_R]).transpose()

acc_all_JP = np.vstack([acc_JPtoFR_N, acc_JPtoJP_N, 
                       acc_JPtoFR_J, acc_JPtoJP_J,
                       acc_JPtoFR_S, acc_JPtoJP_S,
                       acc_JPtoFR_R, acc_JPtoJP_R]).transpose()

acc_all = np.vstack([acc_all_FR, acc_all_JP])


# In[106]:


acc_toFR_N = np.concatenate([acc_FRtoFR_N, acc_JPtoFR_N])
acc_toFR_J = np.concatenate([acc_FRtoFR_J, acc_JPtoFR_J])
acc_toFR_S = np.concatenate([acc_FRtoFR_S, acc_JPtoFR_S])
acc_toFR_R = np.concatenate([acc_FRtoFR_R, acc_JPtoFR_R])

acc_toJP_N = np.concatenate([acc_FRtoJP_N, acc_JPtoJP_N])
acc_toJP_J = np.concatenate([acc_FRtoJP_J, acc_JPtoJP_J])
acc_toJP_S = np.concatenate([acc_FRtoJP_S, acc_JPtoJP_S])
acc_toJP_R = np.concatenate([acc_FRtoJP_R, acc_JPtoJP_R])

#sub_id = np.concatenate([list(range(0,24)), list(range(0,20))])
sub_type = np.concatenate([np.zeros(24), np.ones(20)])

df = pd.DataFrame(
    data =  {'sub_type':sub_type, 'acc_toFR_N': acc_toFR_N, 'acc_toFR_J': acc_toFR_J, 'acc_toFR_S': acc_toFR_S, 'acc_toFR_R':acc_toFR_R,
            'acc_toJP_N':acc_toJP_N, 'acc_toJP_J':acc_toJP_J, 'acc_toJP_S':acc_toJP_S, 'acc_toJP_R':acc_toJP_R})

df.to_csv('Exp2_data_3wayANOVA.csv',  index=False)


# In[ ]:





# In[154]:


acc_N = np.concatenate([acc_toFR_N, acc_toJP_N])
acc_J = np.concatenate([acc_toFR_J, acc_toJP_J])
acc_S = np.concatenate([acc_toFR_S, acc_toJP_S])
acc_R = np.concatenate([acc_toFR_R, acc_toJP_R])

t_score, p_value = ttest_rel(acc_N, acc_J, alternative = 'greater')
cohen_d = compute_effsize(acc_N, acc_J, paired=True, eftype='cohen')
print("Normal vs Jabberwocky")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_N, acc_S, alternative = 'greater')
cohen_d = compute_effsize(acc_N, acc_S, paired=True, eftype='cohen')
print("\nNormal vs Shuffled")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_N, acc_R, alternative = 'greater')
cohen_d = compute_effsize(acc_N, acc_R, paired=True, eftype='cohen')
print("\nNormal vs Reversed")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_J, acc_S, alternative = 'greater')
cohen_d = compute_effsize(acc_J, acc_S, paired=True, eftype='cohen')
print("\nJabberwocky vs Shuffled")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_J, acc_R, alternative = 'greater')
cohen_d = compute_effsize(acc_J, acc_R, paired=True, eftype='cohen')
print("\nJabberwocky vs Reversed")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_S, acc_R, alternative = 'greater')
cohen_d = compute_effsize(acc_S, acc_R, paired=True, eftype='cohen')
print("\nShuffled vs Reversed")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[165]:


acc_FRtoFR = np.mean(np.vstack([acc_FRtoFR_N, acc_FRtoFR_J, acc_FRtoFR_S, acc_FRtoFR_R]), axis=0)
acc_FRtoJP = np.mean(np.vstack([acc_FRtoJP_N, acc_FRtoJP_J, acc_FRtoJP_S, acc_FRtoJP_R]), axis=0)
acc_JPtoFR = np.mean(np.vstack([acc_JPtoFR_N, acc_JPtoFR_J, acc_JPtoFR_S, acc_JPtoFR_R]), axis=0)
acc_JPtoJP = np.mean(np.vstack([acc_JPtoJP_N, acc_JPtoJP_J, acc_JPtoJP_S, acc_JPtoJP_R]), axis=0)

t_score, p_value = ttest_rel(acc_FRtoFR, acc_FRtoJP, alternative = 'greater')
cohen_d = compute_effsize(acc_FRtoFR, acc_FRtoJP, paired=True, eftype='cohen')
print("FR subjects => FR vs JP")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(acc_JPtoJP, acc_JPtoFR, alternative = 'greater')
cohen_d = compute_effsize(acc_JPtoJP, acc_JPtoFR, paired=True, eftype='cohen')
print("\nJP subjects => JP vs FR")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[170]:


print('Mean diff FR sub => FR vs JP: ' +  str(np.mean(acc_FRtoFR) - np.mean(acc_FRtoJP) ))
print('Mean diff JP sub => JP vs FR: ' +  str(np.mean(acc_JPtoJP) - np.mean(acc_JPtoFR) ))


# In[ ]:





# In[ ]:





# In[136]:


###Pitch shift size

target_group = 'FR'
sub_exclude = []
shift_M_FR = []
shift_F_FR = []
for sub_id in range(1,25):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        dat_targ = dat[dat['speakerSex'] == 'M']
        dat_targ = dat_targ.reset_index()
        shift_M_FR.append(dat_targ['shiftSize'][0])
        
        dat_targ = dat[dat['speakerSex'] == 'F']
        dat_targ = dat_targ.reset_index()
        shift_F_FR.append(dat_targ['shiftSize'][0])
        
print('FR subjects => M: ' + str(np.mean(shift_M_FR)))
print('FR subjects => F: ' + str(np.mean(shift_F_FR)))
print('FR subjects => All: ' + str(np.mean(np.concatenate([shift_M_FR,shift_F_FR]))))


# In[135]:


###Pitch shift size

target_group = 'JP'
sub_exclude = [5]
shift_M_JP = []
shift_F_JP = []
for sub_id in range(1,22):

    if sub_id in sub_exclude:
        continue
    else:
        data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp2_{}_raw'.format(target_group)
        data_file = '{}02_{:02d}.csv'.format(target_group, sub_id)
        dat = pd.read_csv( os.path.join(data_dir, data_file) )

        dat_targ = dat[dat['speakerSex'] == 'M']
        dat_targ = dat_targ.reset_index()
        shift_M_JP.append(dat_targ['shiftSize'][0])
        
        dat_targ = dat[dat['speakerSex'] == 'F']
        dat_targ = dat_targ.reset_index()
        shift_F_JP.append(dat_targ['shiftSize'][0])
        
print('JP subjects => M: ' + str(np.mean(shift_M_JP)))
print('JP subjects => F: ' + str(np.mean(shift_F_JP)))
print('JP subjects => All: ' + str(np.mean(np.concatenate([shift_M_JP,shift_F_JP]))))


# In[141]:


shift_all = np.concatenate([shift_M_FR, shift_F_FR, shift_M_JP, shift_F_JP])

subject_type = np.concatenate([np.ones(len(shift_M_FR) * 2), np.zeros(len(shift_M_JP) * 2)])
speaker_sex =  np.concatenate([np.ones(len(shift_M_FR)), np.zeros(len(shift_M_FR)), np.ones(len(shift_M_JP)), np.zeros(len(shift_M_JP))])
sub_id = np.concatenate([list(range(0,len(shift_M_FR))), list(range(0,len(shift_M_FR))),  list(range(len(shift_M_FR),len(shift_M_FR)+len(shift_M_JP))),  list(range(len(shift_M_FR),len(shift_M_FR)+len(shift_M_JP)))])


df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'shift_all': shift_all, 'subject_type': subject_type, 'speaker_sex': speaker_sex})

aov = mixed_anova(dv='shift_all', between='subject_type', within='speaker_sex', subject='sub_id', data=df)
aov.round(5)


# In[144]:


shift_F = np.concatenate([shift_F_FR,shift_F_JP])
shift_M = np.concatenate([shift_M_FR,shift_M_JP])
print('Mean F shiftl: ' + str(np.mean(shift_F)))
print('Mean M shiftl: ' + str(np.mean(shift_M)))


# In[149]:


t_score, p_value = ttest_rel(shift_F, shift_M, alternative = 'greater')
cohen_d = compute_effsize(shift_F, shift_M, paired=True, eftype='cohen')
print("F vs M speaker")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

