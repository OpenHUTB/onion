#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def calc_hitrate(dat):

    rt_threshold = 100
    
    result_bias = []
    result_unbias = []
    result_pi = []
    result_chance = []
    
    conf_matrix = np.zeros((4,4))
    for tt in range(3, len(dat)):
        if dat['responseTime'][tt] < rt_threshold:
            if dat['emoType'][tt][1] == 'h':
                stim_ind = 0         
            elif dat['emoType'][tt][1] == 's':
                stim_ind = 1       
            elif dat['emoType'][tt][1] == 'a':
                stim_ind = 2        
            elif dat['emoType'][tt][1] == 'n':
                stim_ind = 3

            if dat['responseCode'][tt][0] == 'h':
                resp_ind = 0            
            elif dat['responseCode'][tt][0] == 's':
                resp_ind = 1             
            elif dat['responseCode'][tt][0] == 'a':
                resp_ind = 2                         
            elif dat['responseCode'][tt][0] == 'n':            
                resp_ind = 3   

            conf_matrix[stim_ind, resp_ind] = conf_matrix[stim_ind, resp_ind] + 1

    stim_sum = np.sum(conf_matrix, axis=1)
    resp_sum = np.sum(conf_matrix, axis=0)
    total_sum = np.sum(conf_matrix)
        
    result_bias.append( conf_matrix[0, 0] / stim_sum[0] ) #Happy biased
    result_bias.append( conf_matrix[1, 1] / stim_sum[1] ) #Sad biased 
    result_bias.append( conf_matrix[2, 2] / stim_sum[2] ) #Afraid biased
    result_bias.append( conf_matrix[3, 3] / stim_sum[3] ) #Neutral biased
    
    #Proportion index
    p = conf_matrix[0, 0] / stim_sum[0]
    result_pi.append(transforrm_pi(p, 4))
    p = conf_matrix[1, 1] / stim_sum[1]
    result_pi.append(transforrm_pi(p, 4))
    p = conf_matrix[2, 2] / stim_sum[2]
    result_pi.append(transforrm_pi(p, 4))
    p = conf_matrix[3, 3] / stim_sum[3]
    result_pi.append(transforrm_pi(p, 4))
    
    if resp_sum[0] == 0: #Happy unbiased
        result_unbias.append(0)
    else:
        result_unbias.append( conf_matrix[0, 0]*conf_matrix[0, 0] / (stim_sum[0] * resp_sum[0]) )
    if resp_sum[1] == 0: #Sad unbiased 
        result_unbias.append(0)
    else:
        result_unbias.append( conf_matrix[1, 1]*conf_matrix[1, 1] / (stim_sum[1] * resp_sum[1]) )
    if resp_sum[2] == 0: #Afraid unbiased
        result_unbias.append(0) 
    else:
        result_unbias.append( conf_matrix[2, 2]*conf_matrix[2, 2] / (stim_sum[2] * resp_sum[2]) )
    if resp_sum[3] == 0: #Neutral biased
        result_unbias.append(0)
    else:
        result_unbias.append( conf_matrix[3, 3]*conf_matrix[3, 3] / (stim_sum[3] * resp_sum[3]) )

    #Chance propotion
    result_chance.append( stim_sum[0] * resp_sum[0] / (total_sum*total_sum))
    result_chance.append( stim_sum[1] * resp_sum[1] / (total_sum*total_sum))
    result_chance.append( stim_sum[2] * resp_sum[2] / (total_sum*total_sum))
    result_chance.append( stim_sum[3] * resp_sum[3] / (total_sum*total_sum))        
    
    return result_bias, result_pi, result_unbias, result_chance, conf_matrix


# In[3]:


def transforrm_pi(p, k):
    #Rosenthal (1989)
    #p: biased hit rate
    #k: number of choices
    pi = p*(k-1) / (1 + p * (k - 2))
    return pi


# In[ ]:





# In[4]:


target_group = 'FR'
condition_names = ['FR_N'] 
sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoFR_N = happy_unbias
sad_unbias_FRtoFR_N = sad_unbias
afraid_unbias_FRtoFR_N = afraid_unbias
neutral_unbias_FRtoFR_N = neutral_unbias


# In[5]:


print('FR participants => FR normal stimuli...')

print('\nBiased hit rate...')
print('Happy: ' + str(np.mean(happy_bias)))
print('Sad: ' + str(np.mean(sad_bias)))
print('Afraid: ' + str(np.mean(afraid_bias)))
print('Neutral: ' + str(np.mean(neutral_bias)))
print('Mean: ' + str(np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)])))

print('\nProportion index...')
print('Happy: ' + str(transforrm_pi( np.mean(happy_bias), 4)))
print('Sad: ' + str(transforrm_pi( np.mean(sad_bias), 4)))
print('Afraid: ' + str(transforrm_pi( np.mean(afraid_bias), 4)))
print('Neutral: ' + str(transforrm_pi( np.mean(neutral_bias), 4)))
print('Mean: ' + str(transforrm_pi( np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)]), 4)))

print('\nUnbiased hit rate...')
print('Happy: ' + str(np.mean(happy_unbias)))
print('Sad: ' + str(np.mean(sad_unbias)))
print('Afraid: ' + str(np.mean(afraid_unbias)))
print('Neutral: ' + str(np.mean(neutral_unbias)))

print('\nChance proportion...')
print('Happy: ' + str(np.mean(happy_chance)))
print('Sad: ' + str(np.mean(sad_chance)))
print('Afraid: ' + str(np.mean(afraid_chance)))
print('Neutral: ' + str(np.mean(neutral_chance)))

print('\nT values ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[0] ))
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[0] ))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[0] ))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[0] ))

print('\nP values by t-tests, Bonferroni corrected ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[1] * 12)) #Bonferroni correcte (for 12 tests)
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[1] * 12))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[1] * 12))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[1] * 12))


# In[6]:


target_group = 'FR'
condition_names = ['JP_N'] 
sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoJP_N = happy_unbias
sad_unbias_FRtoJP_N = sad_unbias
afraid_unbias_FRtoJP_N = afraid_unbias
neutral_unbias_FRtoJP_N = neutral_unbias


# In[7]:


print('FR participants => JP normal stimuli...')

print('\nBiased hit rate...')
print('Happy: ' + str(np.mean(happy_bias)))
print('Sad: ' + str(np.mean(sad_bias)))
print('Afraid: ' + str(np.mean(afraid_bias)))
print('Neutral: ' + str(np.mean(neutral_bias)))
print('Mean: ' + str(np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)])))

print('\nProportion index...')
print('Happy: ' + str(transforrm_pi( np.mean(happy_bias), 4)))
print('Sad: ' + str(transforrm_pi( np.mean(sad_bias), 4)))
print('Afraid: ' + str(transforrm_pi( np.mean(afraid_bias), 4)))
print('Neutral: ' + str(transforrm_pi( np.mean(neutral_bias), 4)))
print('Mean: ' + str(transforrm_pi( np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)]), 4)))

print('\nUnbiased hit rate...')
print('Happy: ' + str(np.mean(happy_unbias)))
print('Sad: ' + str(np.mean(sad_unbias)))
print('Afraid: ' + str(np.mean(afraid_unbias)))
print('Neutral: ' + str(np.mean(neutral_unbias)))

print('\nChance proportion...')
print('Happy: ' + str(np.mean(happy_chance)))
print('Sad: ' + str(np.mean(sad_chance)))
print('Afraid: ' + str(np.mean(afraid_chance)))
print('Neutral: ' + str(np.mean(neutral_chance)))

print('\nT values ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[0] ))
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[0] ))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[0] ))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[0] ))

print('\nP values by t-tests, Bonferroni corrected ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[1] * 12)) #Bonferroni correcte (for 12 tests)
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[1] * 12))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[1] * 12))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[1] * 12))


# In[20]:


target_group = 'FR'
condition_names = ['SE_N'] 
sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoSE_N = happy_unbias
sad_unbias_FRtoSE_N = sad_unbias
afraid_unbias_FRtoSE_N = afraid_unbias
neutral_unbias_FRtoSE_N = neutral_unbias


# In[21]:


print('FR participants => SE normal stimuli...')

print('\nBiased hit rate...')
print('Happy: ' + str(np.mean(happy_bias)))
print('Sad: ' + str(np.mean(sad_bias)))
print('Afraid: ' + str(np.mean(afraid_bias)))
print('Neutral: ' + str(np.mean(neutral_bias)))
print('Mean: ' + str(np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)])))

print('\nProportion index...')
print('Happy: ' + str(transforrm_pi( np.mean(happy_bias), 4)))
print('Sad: ' + str(transforrm_pi( np.mean(sad_bias), 4)))
print('Afraid: ' + str(transforrm_pi( np.mean(afraid_bias), 4)))
print('Neutral: ' + str(transforrm_pi( np.mean(neutral_bias), 4)))
print('Mean: ' + str(transforrm_pi( np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)]), 4)))

print('\nUnbiased hit rate...')
print('Happy: ' + str(np.mean(happy_unbias)))
print('Sad: ' + str(np.mean(sad_unbias)))
print('Afraid: ' + str(np.mean(afraid_unbias)))
print('Neutral: ' + str(np.mean(neutral_unbias)))

print('\nChance proportion...')
print('Happy: ' + str(np.mean(happy_chance)))
print('Sad: ' + str(np.mean(sad_chance)))
print('Afraid: ' + str(np.mean(afraid_chance)))
print('Neutral: ' + str(np.mean(neutral_chance)))

print('\nT values ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[0] ))
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[0] ))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[0] ))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[0] ))

print('\nP values by t-tests, Bonferroni corrected ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[1] * 12)) #Bonferroni correcte (for 12 tests)
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[1] * 12))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[1] * 12))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[1] * 12))


# In[22]:


target_group = 'JP'
condition_names = ['JP_N'] 
sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoJP_N = happy_unbias
sad_unbias_JPtoJP_N = sad_unbias
afraid_unbias_JPtoJP_N = afraid_unbias
neutral_unbias_JPtoJP_N = neutral_unbias


# In[23]:


print('JP participants => JP normal stimuli...')

print('\nBiased hit rate...')
print('Happy: ' + str(np.mean(happy_bias)))
print('Sad: ' + str(np.mean(sad_bias)))
print('Afraid: ' + str(np.mean(afraid_bias)))
print('Neutral: ' + str(np.mean(neutral_bias)))
print('Mean: ' + str(np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)])))

print('\nProportion index...')
print('Happy: ' + str(transforrm_pi( np.mean(happy_bias), 4)))
print('Sad: ' + str(transforrm_pi( np.mean(sad_bias), 4)))
print('Afraid: ' + str(transforrm_pi( np.mean(afraid_bias), 4)))
print('Neutral: ' + str(transforrm_pi( np.mean(neutral_bias), 4)))
print('Mean: ' + str(transforrm_pi( np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)]), 4)))

print('\nUnbiased hit rate...')
print('Happy: ' + str(np.mean(happy_unbias)))
print('Sad: ' + str(np.mean(sad_unbias)))
print('Afraid: ' + str(np.mean(afraid_unbias)))
print('Neutral: ' + str(np.mean(neutral_unbias)))

print('\nChance proportion...')
print('Happy: ' + str(np.mean(happy_chance)))
print('Sad: ' + str(np.mean(sad_chance)))
print('Afraid: ' + str(np.mean(afraid_chance)))
print('Neutral: ' + str(np.mean(neutral_chance)))

print('\nT values ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[0] ))
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[0] ))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[0] ))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[0] ))

print('\nP values by t-tests, Bonferroni corrected ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[1] * 12)) #Bonferroni correcte (for 12 tests)
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[1] * 12))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[1] * 12))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[1] * 12))


# In[24]:


target_group = 'JP'
condition_names = ['FR_N'] 
sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoFR_N = happy_unbias
sad_unbias_JPtoFR_N = sad_unbias
afraid_unbias_JPtoFR_N = afraid_unbias
neutral_unbias_JPtoFR_N = neutral_unbias


# In[25]:


print('JP participants => FR normal stimuli...')

print('\nBiased hit rate...')
print('Happy: ' + str(np.mean(happy_bias)))
print('Sad: ' + str(np.mean(sad_bias)))
print('Afraid: ' + str(np.mean(afraid_bias)))
print('Neutral: ' + str(np.mean(neutral_bias)))
print('Mean: ' + str(np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)])))

print('\nProportion index...')
print('Happy: ' + str(transforrm_pi( np.mean(happy_bias), 4)))
print('Sad: ' + str(transforrm_pi( np.mean(sad_bias), 4)))
print('Afraid: ' + str(transforrm_pi( np.mean(afraid_bias), 4)))
print('Neutral: ' + str(transforrm_pi( np.mean(neutral_bias), 4)))
print('Mean: ' + str(transforrm_pi( np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)]), 4)))

print('\nUnbiased hit rate...')
print('Happy: ' + str(np.mean(happy_unbias)))
print('Sad: ' + str(np.mean(sad_unbias)))
print('Afraid: ' + str(np.mean(afraid_unbias)))
print('Neutral: ' + str(np.mean(neutral_unbias)))

print('\nChance proportion...')
print('Happy: ' + str(np.mean(happy_chance)))
print('Sad: ' + str(np.mean(sad_chance)))
print('Afraid: ' + str(np.mean(afraid_chance)))
print('Neutral: ' + str(np.mean(neutral_chance)))

print('\nT values ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[0] ))
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[0] ))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[0] ))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[0] ))

print('\nP values by t-tests, Bonferroni corrected ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[1] * 12)) #Bonferroni correcte (for 12 tests)
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[1] * 12))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[1] * 12))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[1] * 12))


# In[26]:


target_group = 'JP'
condition_names = ['SE_N'] 
sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoSE_N = happy_unbias
sad_unbias_JPtoSE_N = sad_unbias
afraid_unbias_JPtoSE_N = afraid_unbias
neutral_unbias_JPtoSE_N = neutral_unbias


# In[27]:


print('JP participants => SE normal stimuli...')

print('\nBiased hit rate...')
print('Happy: ' + str(np.mean(happy_bias)))
print('Sad: ' + str(np.mean(sad_bias)))
print('Afraid: ' + str(np.mean(afraid_bias)))
print('Neutral: ' + str(np.mean(neutral_bias)))
print('Mean: ' + str(np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)])))

print('\nProportion index...')
print('Happy: ' + str(transforrm_pi( np.mean(happy_bias), 4)))
print('Sad: ' + str(transforrm_pi( np.mean(sad_bias), 4)))
print('Afraid: ' + str(transforrm_pi( np.mean(afraid_bias), 4)))
print('Neutral: ' + str(transforrm_pi( np.mean(neutral_bias), 4)))
print('Mean: ' + str(transforrm_pi( np.mean([np.mean(happy_bias), np.mean(sad_bias), np.mean(afraid_bias), np.mean(neutral_bias)]), 4)))

print('\nUnbiased hit rate...')
print('Happy: ' + str(np.mean(happy_unbias)))
print('Sad: ' + str(np.mean(sad_unbias)))
print('Afraid: ' + str(np.mean(afraid_unbias)))
print('Neutral: ' + str(np.mean(neutral_unbias)))

print('\nChance proportion...')
print('Happy: ' + str(np.mean(happy_chance)))
print('Sad: ' + str(np.mean(sad_chance)))
print('Afraid: ' + str(np.mean(afraid_chance)))
print('Neutral: ' + str(np.mean(neutral_chance)))

print('\nT values ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[0] ))
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[0] ))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[0] ))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[0] ))

print('\nP values by t-tests, Bonferroni corrected ...')
print('Happy: ' + str(ttest_ind(happy_unbias, happy_chance, alternative = 'greater')[1] * 12)) #Bonferroni correcte (for 12 tests)
print('Sad: ' + str(ttest_ind(sad_unbias, sad_chance, alternative = 'greater')[1] * 12))
print('Afraid: ' + str(ttest_ind(afraid_unbias, afraid_chance, alternative = 'greater')[1] * 12))
print('Neutral: ' + str(ttest_ind(neutral_unbias, neutral_chance, alternative = 'greater')[1] * 12))


# In[28]:


mean_unbias_FRtoFR_N = np.mean([happy_unbias_FRtoFR_N, sad_unbias_FRtoFR_N, afraid_unbias_FRtoFR_N], axis=0)
mean_unbias_FRtoJP_N = np.mean([happy_unbias_FRtoJP_N, sad_unbias_FRtoJP_N, afraid_unbias_FRtoJP_N], axis=0)
mean_unbias_JPtoJP_N = np.mean([happy_unbias_JPtoJP_N, sad_unbias_JPtoJP_N, afraid_unbias_JPtoJP_N], axis=0)
mean_unbias_JPtoFR_N = np.mean([happy_unbias_JPtoFR_N, sad_unbias_JPtoFR_N, afraid_unbias_JPtoFR_N], axis=0)
mean_unbias_FRtoSE_N = np.mean([happy_unbias_FRtoSE_N, sad_unbias_FRtoSE_N, afraid_unbias_FRtoSE_N], axis=0)
mean_unbias_JPtoSE_N = np.mean([happy_unbias_JPtoSE_N, sad_unbias_JPtoSE_N, afraid_unbias_JPtoSE_N], axis=0)

#mean_unbias_FRtoFR_N = np.mean([happy_unbias_FRtoFR_N, sad_unbias_FRtoFR_N, afraid_unbias_FRtoFR_N, neutral_unbias_FRtoFR_N], axis=0)
#mean_unbias_FRtoJP_N = np.mean([happy_unbias_FRtoJP_N, sad_unbias_FRtoJP_N, afraid_unbias_FRtoJP_N, neutral_unbias_FRtoJP_N], axis=0)
#mean_unbias_JPtoJP_N = np.mean([happy_unbias_JPtoJP_N, sad_unbias_JPtoJP_N, afraid_unbias_JPtoJP_N, neutral_unbias_JPtoJP_N], axis=0)
#mean_unbias_JPtoFR_N = np.mean([happy_unbias_JPtoFR_N, sad_unbias_JPtoFR_N, afraid_unbias_JPtoFR_N, neutral_unbias_JPtoFR_N], axis=0)
#mean_unbias_FRtoSE_N = np.mean([happy_unbias_FRtoSE_N, sad_unbias_FRtoSE_N, afraid_unbias_FRtoSE_N, neutral_unbias_FRtoSE_N], axis=0)
#mean_unbias_JPtoSE_N = np.mean([happy_unbias_JPtoSE_N, sad_unbias_JPtoSE_N, afraid_unbias_JPtoSE_N, neutral_unbias_JPtoSE_N], axis=0)

mean_unbias_all_N = np.concatenate([mean_unbias_FRtoFR_N, mean_unbias_FRtoJP_N, mean_unbias_JPtoJP_N,mean_unbias_JPtoFR_N])
#1...FR, 0...JP
subject_type = np.concatenate([np.ones(len(mean_unbias_FRtoFR_N) * 2), np.zeros(len(mean_unbias_JPtoJP_N) * 2)])
lang_type =  np.concatenate([np.ones(len(mean_unbias_FRtoFR_N)), np.zeros(len(mean_unbias_FRtoFR_N)), np.zeros(len(mean_unbias_JPtoJP_N)), np.ones(len(mean_unbias_JPtoJP_N))])
sub_id = np.concatenate([list(range(0,len(mean_unbias_FRtoFR_N))), list(range(0,len(mean_unbias_FRtoFR_N))),  list(range(len(mean_unbias_FRtoFR_N),len(mean_unbias_FRtoFR_N)+len(mean_unbias_JPtoJP_N))),  list(range(len(mean_unbias_FRtoFR_N),len(mean_unbias_FRtoFR_N)+len(mean_unbias_JPtoJP_N)))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': mean_unbias_all_N, 'subject_type': subject_type, 'lang_type': lang_type})


# In[29]:


fig = interaction_plot(df.subject_type, df.lang_type, df.hit_rate, colors=['red', 'blue'])
aov = mixed_anova(dv='hit_rate', between='subject_type', within='lang_type', subject='sub_id', data=df)
aov.round(5)


# In[30]:


t_score, p_value = ttest_rel(mean_unbias_FRtoFR_N, mean_unbias_FRtoJP_N, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_FRtoFR_N, mean_unbias_FRtoJP_N, paired=True, eftype='cohen')
print("FR sub => FR vs JP stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(mean_unbias_JPtoJP_N, mean_unbias_JPtoFR_N, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_JPtoJP_N, mean_unbias_JPtoFR_N, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(mean_unbias_FRtoFR_N, mean_unbias_FRtoSE_N, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_FRtoFR_N, mean_unbias_FRtoSE_N, paired=True, eftype='cohen')
print("\nFR sub => FR vs SE stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(mean_unbias_JPtoJP_N, mean_unbias_JPtoSE_N, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_JPtoJP_N, mean_unbias_JPtoSE_N, paired=True, eftype='cohen')
print("\nJP sub => JP vs SE stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[ ]:





# In[31]:


target_group = 'FR'
condition_names = ['FR_J'] 

sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoFR_J = happy_unbias
sad_unbias_FRtoFR_J = sad_unbias
afraid_unbias_FRtoFR_J = afraid_unbias
neutral_unbias_FRtoFR_J = neutral_unbias


# In[32]:


target_group = 'FR'
#condition_names = ['FR_N', 'FR_J', 'FR_S', 'rFR_N'] 
condition_names = ['JP_J'] 


sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoJP_J = happy_unbias
sad_unbias_FRtoJP_J = sad_unbias
afraid_unbias_FRtoJP_J = afraid_unbias
neutral_unbias_FRtoJP_J = neutral_unbias


# In[33]:


target_group = 'JP'
#condition_names = ['FR_N', 'FR_J', 'FR_S', 'rFR_N'] 
condition_names = ['JP_J'] 

sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoJP_J = happy_unbias
sad_unbias_JPtoJP_J = sad_unbias
afraid_unbias_JPtoJP_J = afraid_unbias
neutral_unbias_JPtoJP_J = neutral_unbias


# In[34]:


target_group = 'JP'
condition_names = ['FR_J'] 

sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoFR_J = happy_unbias
sad_unbias_JPtoFR_J = sad_unbias
afraid_unbias_JPtoFR_J = afraid_unbias
neutral_unbias_JPtoFR_J = neutral_unbias


# In[35]:


mean_unbias_FRtoFR_J = np.mean([happy_unbias_FRtoFR_J, sad_unbias_FRtoFR_J, afraid_unbias_FRtoFR_J], axis=0)
mean_unbias_FRtoJP_J = np.mean([happy_unbias_FRtoJP_J, sad_unbias_FRtoJP_J, afraid_unbias_FRtoJP_J], axis=0)
mean_unbias_JPtoJP_J = np.mean([happy_unbias_JPtoJP_J, sad_unbias_JPtoJP_J, afraid_unbias_JPtoJP_J], axis=0)
mean_unbias_JPtoFR_J= np.mean([happy_unbias_JPtoFR_J, sad_unbias_JPtoFR_J, afraid_unbias_JPtoFR_J], axis=0)

#mean_unbias_FRtoFR_J = np.mean([happy_unbias_FRtoFR_J, sad_unbias_FRtoFR_J, afraid_unbias_FRtoFR_J, neutral_unbias_FRtoFR_J], axis=0)
#mean_unbias_FRtoJP_J = np.mean([happy_unbias_FRtoJP_J, sad_unbias_FRtoJP_J, afraid_unbias_FRtoJP_J, neutral_unbias_FRtoJP_J], axis=0)
#mean_unbias_JPtoJP_J = np.mean([happy_unbias_JPtoJP_J, sad_unbias_JPtoJP_J, afraid_unbias_JPtoJP_J, neutral_unbias_JPtoJP_J], axis=0)
#mean_unbias_JPtoFR_J= np.mean([happy_unbias_JPtoFR_J, sad_unbias_JPtoFR_J, afraid_unbias_JPtoFR_J, neutral_unbias_JPtoFR_J], axis=0)


mean_unbias_all_J = np.concatenate([mean_unbias_FRtoFR_J, mean_unbias_FRtoJP_J, mean_unbias_JPtoJP_J, mean_unbias_JPtoFR_J])
#1...FR, 0...JP
subject_type = np.concatenate([np.ones(len(mean_unbias_FRtoFR_N) * 2), np.zeros(len(mean_unbias_JPtoJP_N) * 2)])
lang_type =  np.concatenate([np.ones(len(mean_unbias_FRtoFR_N)), np.zeros(len(mean_unbias_FRtoFR_N)), np.zeros(len(mean_unbias_JPtoJP_N)), np.ones(len(mean_unbias_JPtoJP_N))])
sub_id = np.concatenate([list(range(0,len(mean_unbias_FRtoFR_N))), list(range(0,len(mean_unbias_FRtoFR_N))),  list(range(len(mean_unbias_FRtoFR_N),len(mean_unbias_FRtoFR_N)+len(mean_unbias_JPtoJP_N))),  list(range(len(mean_unbias_FRtoFR_N),len(mean_unbias_FRtoFR_N)+len(mean_unbias_JPtoJP_N)))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': mean_unbias_all_J, 'subject_type': subject_type, 'lang_type': lang_type})


# In[36]:


fig = interaction_plot(df.subject_type, df.lang_type, df.hit_rate, colors=['red', 'blue'])
aov = mixed_anova(dv='hit_rate', between='subject_type', within='lang_type', subject='sub_id', data=df)
aov.round(5)


# In[37]:


t_score, p_value = ttest_rel(mean_unbias_FRtoFR_J, mean_unbias_FRtoJP_J, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_FRtoFR_J, mean_unbias_FRtoJP_J, paired=True, eftype='cohen')
print("FR sub => FR vs JP stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(mean_unbias_JPtoJP_J, mean_unbias_JPtoFR_J, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_JPtoJP_J, mean_unbias_JPtoFR_J, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[ ]:





# In[38]:


target_group = 'FR'
condition_names = ['FR_S'] 

sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoFR_S = happy_unbias
sad_unbias_FRtoFR_S = sad_unbias
afraid_unbias_FRtoFR_S = afraid_unbias
neutral_unbias_FRtoFR_S = neutral_unbias


# In[39]:


target_group = 'FR'
#condition_names = ['FR_N', 'FR_J', 'FR_S', 'rFR_N'] 
condition_names = ['JP_S'] 


sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoJP_S = happy_unbias
sad_unbias_FRtoJP_S = sad_unbias
afraid_unbias_FRtoJP_S = afraid_unbias
neutral_unbias_FRtoJP_S = neutral_unbias


# In[40]:


target_group = 'JP'
#condition_names = ['FR_N', 'FR_J', 'FR_S', 'rFR_N'] 
condition_names = ['JP_S'] 

sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoJP_S = happy_unbias
sad_unbias_JPtoJP_S = sad_unbias
afraid_unbias_JPtoJP_S = afraid_unbias
neutral_unbias_JPtoJP_S = neutral_unbias


# In[41]:


target_group = 'JP'
condition_names = ['FR_S'] 
sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoFR_S = happy_unbias
sad_unbias_JPtoFR_S = sad_unbias
afraid_unbias_JPtoFR_S = afraid_unbias
neutral_unbias_JPtoFR_S = neutral_unbias


# In[42]:


mean_unbias_FRtoFR_S = np.mean([happy_unbias_FRtoFR_S, sad_unbias_FRtoFR_S, afraid_unbias_FRtoFR_S], axis=0)
mean_unbias_FRtoJP_S = np.mean([happy_unbias_FRtoJP_S, sad_unbias_FRtoJP_S, afraid_unbias_FRtoJP_S], axis=0)
mean_unbias_JPtoJP_S = np.mean([happy_unbias_JPtoJP_S, sad_unbias_JPtoJP_S, afraid_unbias_JPtoJP_S], axis=0)
mean_unbias_JPtoFR_S = np.mean([happy_unbias_JPtoFR_S, sad_unbias_JPtoFR_S, afraid_unbias_JPtoFR_S], axis=0)

#mean_unbias_FRtoFR_S = np.mean([happy_unbias_FRtoFR_S, sad_unbias_FRtoFR_S, afraid_unbias_FRtoFR_S, neutral_unbias_FRtoFR_S], axis=0)
#mean_unbias_FRtoJP_S = np.mean([happy_unbias_FRtoJP_S, sad_unbias_FRtoJP_S, afraid_unbias_FRtoJP_S, neutral_unbias_FRtoJP_S], axis=0)
#mean_unbias_JPtoJP_S = np.mean([happy_unbias_JPtoJP_S, sad_unbias_JPtoJP_S, afraid_unbias_JPtoJP_S, neutral_unbias_JPtoJP_S], axis=0)
#mean_unbias_JPtoFR_S = np.mean([happy_unbias_JPtoFR_S, sad_unbias_JPtoFR_S, afraid_unbias_JPtoFR_S, neutral_unbias_JPtoFR_S], axis=0)

mean_unbias_all_S = np.concatenate([mean_unbias_FRtoFR_S, mean_unbias_FRtoJP_S, mean_unbias_JPtoJP_S, mean_unbias_JPtoFR_S])
#1...FR, 0...JP
subject_type = np.concatenate([np.ones(len(mean_unbias_FRtoFR_N) * 2), np.zeros(len(mean_unbias_JPtoJP_N) * 2)])
lang_type =  np.concatenate([np.ones(len(mean_unbias_FRtoFR_N)), np.zeros(len(mean_unbias_FRtoFR_N)), np.zeros(len(mean_unbias_JPtoJP_N)), np.ones(len(mean_unbias_JPtoJP_N))])
sub_id = np.concatenate([list(range(0,len(mean_unbias_FRtoFR_N))), list(range(0,len(mean_unbias_FRtoFR_N))),  list(range(len(mean_unbias_FRtoFR_N),len(mean_unbias_FRtoFR_N)+len(mean_unbias_JPtoJP_N))),  list(range(len(mean_unbias_FRtoFR_N),len(mean_unbias_FRtoFR_N)+len(mean_unbias_JPtoJP_N)))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': mean_unbias_all_S, 'subject_type': subject_type, 'lang_type': lang_type})


# In[43]:


fig = interaction_plot(df.subject_type, df.lang_type, df.hit_rate, colors=['red', 'blue'])
aov = mixed_anova(dv='hit_rate', between='subject_type', within='lang_type', subject='sub_id', data=df)
aov.round(5)


# In[44]:


t_score, p_value = ttest_rel(mean_unbias_FRtoFR_S, mean_unbias_FRtoJP_S, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_FRtoFR_S, mean_unbias_FRtoJP_S, paired=True, eftype='cohen')
print("FR sub => FR vs JP stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(mean_unbias_JPtoJP_S, mean_unbias_JPtoFR_S, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_JPtoJP_S, mean_unbias_JPtoFR_S, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[ ]:





# In[45]:


target_group = 'FR'
condition_names = ['rFR_N'] 
sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoFR_R = happy_unbias
sad_unbias_FRtoFR_R = sad_unbias
afraid_unbias_FRtoFR_R = afraid_unbias
neutral_unbias_FRtoFR_R = neutral_unbias


# In[46]:


target_group = 'FR'
#condition_names = ['FR_N', 'FR_J', 'FR_S', 'rFR_N'] 
condition_names = ['rJP_N'] 


sub_exclude = [10, 15]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_FRtoJP_R = happy_unbias
sad_unbias_FRtoJP_R = sad_unbias
afraid_unbias_FRtoJP_R = afraid_unbias
neutral_unbias_FRtoJP_R = neutral_unbias


# In[47]:


target_group = 'JP'
#condition_names = ['FR_N', 'FR_J', 'FR_S', 'rFR_N'] 
condition_names = ['rJP_N'] 

sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoJP_R = happy_unbias
sad_unbias_JPtoJP_R = sad_unbias
afraid_unbias_JPtoJP_R = afraid_unbias
neutral_unbias_JPtoJP_R = neutral_unbias


# In[48]:


target_group = 'JP'
condition_names = ['rFR_N'] 
sub_exclude = [9]

happy_bias = []
happy_pi = []
happy_unbias = []
happy_chance = []
sad_bias = []
sad_pi = []
sad_unbias = []
sad_chance = []
afraid_bias = []
afraid_pi = []
afraid_unbias = []
afraid_chance = []
neutral_bias = []
neutral_pi = []
neutral_unbias = []
neutral_chance = []
tconf_matrix = np.zeros(4)

for cc in range(len(condition_names)):
    condition_name = condition_names[cc]
    for sub_id in range(1,23):

        if sub_id in sub_exclude:
            continue
        else:
            data_dir = '/Users/nakai-tomoya/Desktop/LFE/Exp1_{}_raw'.format(target_group)
            data_file = '{}01_{:02d}.csv'.format(target_group, sub_id)
            dat = pd.read_csv( os.path.join(data_dir, data_file) )

            dat_targ = dat[dat['condition'] == condition_name]
            dat_targ = dat_targ.reset_index()
            result_bias, result_pi, result_unbias, result_chance, conf_matrix = calc_hitrate(dat_targ)

            happy_bias.append(result_bias[0])
            happy_pi.append(result_pi[0])
            happy_unbias.append(result_unbias[0])    
            happy_chance.append(result_chance[0])   
            
            sad_bias.append(result_bias[1])
            sad_pi.append(result_pi[1])
            sad_unbias.append(result_unbias[1])    
            sad_chance.append(result_chance[1]) 

            afraid_bias.append(result_bias[2])
            afraid_pi.append(result_pi[2])
            afraid_unbias.append(result_unbias[2])    
            afraid_chance.append(result_chance[2]) 

            neutral_bias.append(result_bias[3])
            neutral_pi.append(result_pi[3])
            neutral_unbias.append(result_unbias[3])    
            neutral_chance.append(result_chance[3]) 
            
            tconf_matrix = tconf_matrix + conf_matrix
            
happy_unbias_JPtoFR_R = happy_unbias
sad_unbias_JPtoFR_R = sad_unbias
afraid_unbias_JPtoFR_R = afraid_unbias
neutral_unbias_JPtoFR_R = neutral_unbias


# In[49]:


mean_unbias_FRtoFR_R = np.mean([happy_unbias_FRtoFR_R, sad_unbias_FRtoFR_R, afraid_unbias_FRtoFR_R], axis=0)
mean_unbias_FRtoJP_R = np.mean([happy_unbias_FRtoJP_R, sad_unbias_FRtoJP_R, afraid_unbias_FRtoJP_R], axis=0)
mean_unbias_JPtoJP_R = np.mean([happy_unbias_JPtoJP_R, sad_unbias_JPtoJP_R, afraid_unbias_JPtoJP_R], axis=0)
mean_unbias_JPtoFR_R = np.mean([happy_unbias_JPtoFR_R, sad_unbias_JPtoFR_R, afraid_unbias_JPtoFR_R], axis=0)

#mean_unbias_FRtoFR_R = np.mean([happy_unbias_FRtoFR_R, sad_unbias_FRtoFR_R, afraid_unbias_FRtoFR_R, neutral_unbias_FRtoFR_R], axis=0)
#mean_unbias_FRtoJP_R = np.mean([happy_unbias_FRtoJP_R, sad_unbias_FRtoJP_R, afraid_unbias_FRtoJP_R, neutral_unbias_FRtoJP_R], axis=0)
#mean_unbias_JPtoJP_R = np.mean([happy_unbias_JPtoJP_R, sad_unbias_JPtoJP_R, afraid_unbias_JPtoJP_R, neutral_unbias_JPtoJP_R], axis=0)
#mean_unbias_JPtoFR_R = np.mean([happy_unbias_JPtoFR_R, sad_unbias_JPtoFR_R, afraid_unbias_JPtoFR_R, neutral_unbias_JPtoFR_R], axis=0)


mean_unbias_all_R = np.concatenate([mean_unbias_FRtoFR_R, mean_unbias_FRtoJP_R, mean_unbias_JPtoJP_R, mean_unbias_JPtoFR_R])
#1...FR, 0...JP
subject_type = np.concatenate([np.ones(len(mean_unbias_FRtoFR_N) * 2), np.zeros(len(mean_unbias_JPtoJP_N) * 2)])
lang_type =  np.concatenate([np.ones(len(mean_unbias_FRtoFR_N)), np.zeros(len(mean_unbias_FRtoFR_N)), np.zeros(len(mean_unbias_JPtoJP_N)), np.ones(len(mean_unbias_JPtoJP_N))])
sub_id = np.concatenate([list(range(0,len(mean_unbias_FRtoFR_N))), list(range(0,len(mean_unbias_FRtoFR_N))),  list(range(len(mean_unbias_FRtoFR_N),len(mean_unbias_FRtoFR_N)+len(mean_unbias_JPtoJP_N))),  list(range(len(mean_unbias_FRtoFR_N),len(mean_unbias_FRtoFR_N)+len(mean_unbias_JPtoJP_N)))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': mean_unbias_all_R, 'subject_type': subject_type, 'lang_type': lang_type})


# In[50]:


fig = interaction_plot(df.subject_type, df.lang_type, df.hit_rate, colors=['red', 'blue'])
aov = mixed_anova(dv='hit_rate', between='subject_type', within='lang_type', subject='sub_id', data=df)
aov.round(5)


# In[51]:


t_score, p_value = ttest_rel(mean_unbias_FRtoFR_R, mean_unbias_FRtoJP_R, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_FRtoFR_R, mean_unbias_FRtoJP_R, paired=True, eftype='cohen')
print("FR sub => FR vs JP stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(mean_unbias_JPtoJP_R, mean_unbias_JPtoFR_R, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_JPtoJP_R, mean_unbias_JPtoFR_R, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


# In[ ]:





# In[71]:


mean_unbias_FRtoFR_Happy = np.mean([happy_unbias_FRtoFR_N, happy_unbias_FRtoFR_J, happy_unbias_FRtoFR_S, happy_unbias_FRtoFR_R], axis=0)
mean_unbias_FRtoFR_Sad = np.mean([sad_unbias_FRtoFR_N, sad_unbias_FRtoFR_J, sad_unbias_FRtoFR_S, sad_unbias_FRtoFR_R], axis=0)
mean_unbias_FRtoFR_Afraid = np.mean([afraid_unbias_FRtoFR_N, afraid_unbias_FRtoFR_J, afraid_unbias_FRtoFR_S, afraid_unbias_FRtoFR_R], axis=0)
mean_unbias_FRtoFR_Neutral = np.mean([neutral_unbias_FRtoFR_N, neutral_unbias_FRtoFR_J, neutral_unbias_FRtoFR_S, neutral_unbias_FRtoFR_R], axis=0)

mean_unbias_FRtoJP_Happy = np.mean([happy_unbias_FRtoJP_N, happy_unbias_FRtoJP_J, happy_unbias_FRtoJP_S, happy_unbias_FRtoJP_R], axis=0)
mean_unbias_FRtoJP_Sad = np.mean([sad_unbias_FRtoJP_N, sad_unbias_FRtoJP_J, sad_unbias_FRtoJP_S, sad_unbias_FRtoJP_R], axis=0)
mean_unbias_FRtoJP_Afraid = np.mean([afraid_unbias_FRtoJP_N, afraid_unbias_FRtoJP_J, afraid_unbias_FRtoJP_S, afraid_unbias_FRtoJP_R], axis=0)
mean_unbias_FRtoJP_Neutral = np.mean([neutral_unbias_FRtoJP_N, neutral_unbias_FRtoJP_J, neutral_unbias_FRtoJP_S, neutral_unbias_FRtoJP_R], axis=0)

mean_unbias_JPtoFR_Happy = np.mean([happy_unbias_JPtoFR_N, happy_unbias_JPtoFR_J, happy_unbias_JPtoFR_S, happy_unbias_JPtoFR_R], axis=0)
mean_unbias_JPtoFR_Sad = np.mean([sad_unbias_JPtoFR_N, sad_unbias_JPtoFR_J, sad_unbias_JPtoFR_S, sad_unbias_JPtoFR_R], axis=0)
mean_unbias_JPtoFR_Afraid = np.mean([afraid_unbias_JPtoFR_N, afraid_unbias_JPtoFR_J, afraid_unbias_JPtoFR_S, afraid_unbias_JPtoFR_R], axis=0)
mean_unbias_JPtoFR_Neutral = np.mean([neutral_unbias_JPtoFR_N, neutral_unbias_JPtoFR_J, neutral_unbias_JPtoFR_S, neutral_unbias_JPtoFR_R], axis=0)

mean_unbias_JPtoJP_Happy = np.mean([happy_unbias_JPtoJP_N, happy_unbias_JPtoJP_J, happy_unbias_JPtoJP_S, happy_unbias_JPtoJP_R], axis=0)
mean_unbias_JPtoJP_Sad = np.mean([sad_unbias_JPtoJP_N, sad_unbias_JPtoJP_J, sad_unbias_JPtoJP_S, sad_unbias_JPtoJP_R], axis=0)
mean_unbias_JPtoJP_Afraid = np.mean([afraid_unbias_JPtoJP_N, afraid_unbias_JPtoJP_J, afraid_unbias_JPtoJP_S, afraid_unbias_JPtoJP_R], axis=0)
mean_unbias_JPtoJP_Neutral = np.mean([neutral_unbias_JPtoJP_N, neutral_unbias_JPtoJP_J, neutral_unbias_JPtoJP_S, neutral_unbias_JPtoJP_R], axis=0)


# In[56]:


mean_unbias_all_FR = np.concatenate([mean_unbias_FRtoFR_Happy, mean_unbias_FRtoFR_Sad, mean_unbias_FRtoFR_Afraid, 
                                     mean_unbias_FRtoJP_Happy, mean_unbias_FRtoJP_Sad, mean_unbias_FRtoJP_Afraid])

N_FR = len(mean_unbias_FRtoFR_N)
#1...FR, 0...JP
emo_type = np.concatenate([np.ones(N_FR), 
                           np.ones(N_FR)*2, 
                           np.ones(N_FR)*3, 
                           np.ones(N_FR),
                           np.ones(N_FR)*2, 
                           np.ones(N_FR)*3])

lang_type =  np.concatenate([np.zeros(N_FR), 
                             np.zeros(N_FR), 
                             np.zeros(N_FR), 
                             np.ones(N_FR),
                             np.ones(N_FR),                            
                             np.ones(N_FR)])

sub_id = np.concatenate([list(range(0,N_FR)),
                         list(range(0,N_FR)),
                         list(range(0,N_FR)),
                         list(range(0,N_FR)),
                         list(range(0,N_FR)),
                         list(range(0,N_FR))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': mean_unbias_all_FR, 'emo_type': emo_type, 'lang_type': lang_type})

fig = interaction_plot(df.emo_type, df.lang_type, df.hit_rate, colors=['red', 'blue'])
aov = rm_anova(dv='hit_rate', within=['emo_type','lang_type'], subject='sub_id', data=df, effsize='np2')
aov.round(5)


# In[57]:


mean_unbias_all_JP = np.concatenate([mean_unbias_JPtoJP_Happy, mean_unbias_JPtoJP_Sad, mean_unbias_JPtoJP_Afraid, 
                                     mean_unbias_JPtoFR_Happy, mean_unbias_JPtoFR_Sad, mean_unbias_JPtoFR_Afraid])

N_JP = len(mean_unbias_JPtoJP_N)
#1...FR, 0...JP
emo_type = np.concatenate([np.ones(N_JP), 
                           np.ones(N_JP)*2, 
                           np.ones(N_JP)*3, 
                           np.ones(N_JP),
                           np.ones(N_JP)*2, 
                           np.ones(N_JP)*3])

lang_type =  np.concatenate([np.zeros(N_JP), 
                             np.zeros(N_JP), 
                             np.zeros(N_JP), 
                             np.ones(N_JP),
                             np.ones(N_JP),                            
                             np.ones(N_JP)])

sub_id = np.concatenate([list(range(0,N_JP)),
                         list(range(0,N_JP)),
                         list(range(0,N_JP)),
                         list(range(0,N_JP)),
                         list(range(0,N_JP)),
                         list(range(0,N_JP))])

df = pd.DataFrame(
    data =  {'sub_id':sub_id, 'hit_rate': mean_unbias_all_JP, 'emo_type': emo_type, 'lang_type': lang_type})

fig = interaction_plot(df.emo_type, df.lang_type, df.hit_rate, colors=['red', 'blue'])
aov = rm_anova(dv='hit_rate', within=['emo_type','lang_type'], subject='sub_id', data=df, effsize='np2')
aov.round(5)


# In[371]:


t_score, p_value = ttest_rel(mean_unbias_JPtoJP_Happy, mean_unbias_JPtoFR_Happy, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_JPtoJP_Happy, mean_unbias_JPtoFR_Happy, paired=True, eftype='cohen')
print("JP sub => JP vs FR stim, Happy")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))

t_score, p_value = ttest_rel(mean_unbias_JPtoJP_Sad, mean_unbias_JPtoFR_Sad, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_JPtoJP_Sad, mean_unbias_JPtoFR_Sad, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim, Sad")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))


t_score, p_value = ttest_rel(mean_unbias_JPtoJP_Afraid, mean_unbias_JPtoFR_Afraid, alternative = 'greater')
cohen_d = compute_effsize(mean_unbias_JPtoJP_Afraid, mean_unbias_JPtoFR_Afraid, paired=True, eftype='cohen')
print("\nJP sub => JP vs FR stim, Afraid")
print("t score: " + str(t_score))
print("p value: " + str(p_value))
print("Cohen's d: " + str(cohen_d))



# In[372]:


mean_FR = [np.mean(mean_unbias_FRtoFR_N)*100, np.mean(mean_unbias_JPtoFR_N)*100]
mean_JP = [np.mean(mean_unbias_FRtoJP_N)*100, np.mean(mean_unbias_JPtoJP_N)*100]
mean_SE = [np.mean(mean_unbias_FRtoSE_N)*100, np.mean(mean_unbias_JPtoSE_N)*100]

ci_FRtoFR = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoFR_N*100)-1, loc=np.mean(mean_unbias_FRtoFR_N*100), scale=st.sem(mean_unbias_FRtoFR_N*100))
ci_JPtoFR = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoFR_N*100)-1, loc=np.mean(mean_unbias_JPtoFR_N*100), scale=st.sem(mean_unbias_JPtoFR_N*100))
ci_FRtoJP = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoJP_N*100)-1, loc=np.mean(mean_unbias_FRtoJP_N*100), scale=st.sem(mean_unbias_FRtoJP_N*100))
ci_JPtoJP = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoJP_N*100)-1, loc=np.mean(mean_unbias_JPtoJP_N*100), scale=st.sem(mean_unbias_JPtoJP_N*100))
ci_FRtoSE = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoSE_N*100)-1, loc=np.mean(mean_unbias_FRtoSE_N*100), scale=st.sem(mean_unbias_FRtoSE_N*100))
ci_JPtoSE = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoSE_N*100)-1, loc=np.mean(mean_unbias_JPtoSE_N*100), scale=st.sem(mean_unbias_JPtoSE_N*100))


xpos_FR = np.array([0, 1])
xpos_JP = np.array([0.3, 1.3])
xpos_SE = np.array([0.6, 1.6])
xpos_label = np.array([0.3, 1.3])
width = 0.3
labels = ['FR', 'JP']

figure, ax = plt.subplots(figsize=[3,4])
ax.bar(xpos_FR, mean_FR, color=[1,1,1], edgecolor ='black', width=width, align='center')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FRtoFR, color=[0.3,0.3,0.3])
ax.plot([xpos_FR[1],xpos_FR[1]], ci_JPtoFR, color=[0.3,0.3,0.3])
ax.bar(xpos_JP, mean_JP, color=[0.5,0.5,0.5], edgecolor ='black', width=width, align='center')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_FRtoJP, color=[0.3,0.3,0.3])
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JPtoJP, color=[0.3,0.3,0.3])
ax.bar(xpos_SE, mean_SE, color=[0,0,0], edgecolor ='black', width=width, align='center')
ax.plot([xpos_SE[0],xpos_SE[0]], ci_FRtoSE, color=[0.3,0.3,0.3])
ax.plot([xpos_SE[1],xpos_SE[1]], ci_JPtoSE, color=[0.3,0.3,0.3])

plt.xlabel('Participant group', fontname='Arial')
plt.ylabel('Unbiased hit rate (%)', fontname='Arial')
plt.xticks(xpos_label, labels, fontname='Arial')
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial')
plt.tight_layout()

plt.show()
fname = 'Exp1_Barplot_Normal.pdf'
figure.savefig(fname, format='pdf')


# In[373]:


mean_FR = [np.mean(mean_unbias_FRtoFR_J)*100, np.mean(mean_unbias_JPtoFR_J)*100]
mean_JP = [np.mean(mean_unbias_FRtoJP_J)*100, np.mean(mean_unbias_JPtoJP_J)*100]

ci_FRtoFR = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoFR_J*100)-1, loc=np.mean(mean_unbias_FRtoFR_J*100), scale=st.sem(mean_unbias_FRtoFR_J*100))
ci_JPtoFR = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoFR_J*100)-1, loc=np.mean(mean_unbias_JPtoFR_J*100), scale=st.sem(mean_unbias_JPtoFR_J*100))
ci_FRtoJP = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoJP_J*100)-1, loc=np.mean(mean_unbias_FRtoJP_J*100), scale=st.sem(mean_unbias_FRtoJP_J*100))
ci_JPtoJP = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoJP_J*100)-1, loc=np.mean(mean_unbias_JPtoJP_J*100), scale=st.sem(mean_unbias_JPtoJP_J*100))


xpos_FR = np.array([0, 0.7])
xpos_JP = np.array([0.3, 1])
xpos_label = np.array([0.15, 0.85])
width = 0.3
labels = ['FR', 'JP']

figure, ax = plt.subplots(figsize=[3,4])
ax.bar(xpos_FR, mean_FR, color=[1,1,1], edgecolor ='black', width=width, align='center')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FRtoFR, color=[0.3,0.3,0.3])
ax.plot([xpos_FR[1],xpos_FR[1]], ci_JPtoFR, color=[0.3,0.3,0.3])
ax.bar(xpos_JP, mean_JP, color=[0.5,0.5,0.5], edgecolor ='black', width=width, align='center')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_FRtoJP, color=[0.3,0.3,0.3])
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JPtoJP, color=[0.3,0.3,0.3])


plt.xlabel('Participant group', fontname='Arial')
plt.ylabel('Unbiased hit rate (%)', fontname='Arial')
plt.xticks(xpos_label, labels, fontname='Arial')
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial')
plt.tight_layout()

plt.show()
fname = 'Exp1_Barplot_Jabberwocky.pdf'
figure.savefig(fname, format='pdf')


# In[374]:


mean_FR = [np.mean(mean_unbias_FRtoFR_S)*100, np.mean(mean_unbias_JPtoFR_S)*100]
mean_JP = [np.mean(mean_unbias_FRtoJP_S)*100, np.mean(mean_unbias_JPtoJP_S)*100]

ci_FRtoFR = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoFR_S*100)-1, loc=np.mean(mean_unbias_FRtoFR_S*100), scale=st.sem(mean_unbias_FRtoFR_S*100))
ci_JPtoFR = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoFR_S*100)-1, loc=np.mean(mean_unbias_JPtoFR_S*100), scale=st.sem(mean_unbias_JPtoFR_S*100))
ci_FRtoJP = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoJP_S*100)-1, loc=np.mean(mean_unbias_FRtoJP_S*100), scale=st.sem(mean_unbias_FRtoJP_S*100))
ci_JPtoJP = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoJP_S*100)-1, loc=np.mean(mean_unbias_JPtoJP_S*100), scale=st.sem(mean_unbias_JPtoJP_S*100))


xpos_FR = np.array([0, 0.7])
xpos_JP = np.array([0.3, 1])
xpos_label = np.array([0.15, 0.85])
width = 0.3
labels = ['FR', 'JP']

figure, ax = plt.subplots(figsize=[3,4])
ax.bar(xpos_FR, mean_FR, color=[1,1,1], edgecolor ='black', width=width, align='center')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FRtoFR, color=[0.3,0.3,0.3])
ax.plot([xpos_FR[1],xpos_FR[1]], ci_JPtoFR, color=[0.3,0.3,0.3])
ax.bar(xpos_JP, mean_JP, color=[0.5,0.5,0.5], edgecolor ='black', width=width, align='center')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_FRtoJP, color=[0.3,0.3,0.3])
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JPtoJP, color=[0.3,0.3,0.3])


plt.xlabel('Participant group', fontname='Arial')
plt.ylabel('Unbiased hit rate (%)', fontname='Arial')
plt.xticks(xpos_label, labels, fontname='Arial')
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial')
plt.tight_layout()

plt.show()
fname = 'Exp1_Barplot_Shuffled.pdf'
figure.savefig(fname, format='pdf')


# In[375]:


mean_FR = [np.mean(mean_unbias_FRtoFR_R)*100, np.mean(mean_unbias_JPtoFR_R)*100]
mean_JP = [np.mean(mean_unbias_FRtoJP_R)*100, np.mean(mean_unbias_JPtoJP_R)*100]

ci_FRtoFR = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoFR_R*100)-1, loc=np.mean(mean_unbias_FRtoFR_R*100), scale=st.sem(mean_unbias_FRtoFR_R*100))
ci_JPtoFR = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoFR_R*100)-1, loc=np.mean(mean_unbias_JPtoFR_R*100), scale=st.sem(mean_unbias_JPtoFR_R*100))
ci_FRtoJP = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoJP_R*100)-1, loc=np.mean(mean_unbias_FRtoJP_R*100), scale=st.sem(mean_unbias_FRtoJP_R*100))
ci_JPtoJP = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoJP_R*100)-1, loc=np.mean(mean_unbias_JPtoJP_R*100), scale=st.sem(mean_unbias_JPtoJP_R*100))


xpos_FR = np.array([0, 0.7])
xpos_JP = np.array([0.3, 1])
xpos_label = np.array([0.15, 0.85])
width = 0.3
labels = ['FR', 'JP']

figure, ax = plt.subplots(figsize=[3,4])
ax.bar(xpos_FR, mean_FR, color=[1,1,1], edgecolor ='black', width=width, align='center')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FRtoFR, color=[0.3,0.3,0.3])
ax.plot([xpos_FR[1],xpos_FR[1]], ci_JPtoFR, color=[0.3,0.3,0.3])
ax.bar(xpos_JP, mean_JP, color=[0.5,0.5,0.5], edgecolor ='black', width=width, align='center')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_FRtoJP, color=[0.3,0.3,0.3])
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JPtoJP, color=[0.3,0.3,0.3])


plt.xlabel('Participant group', fontname='Arial')
plt.ylabel('Unbiased hit rate (%)', fontname='Arial')
plt.xticks(xpos_label, labels, fontname='Arial')
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial')
plt.tight_layout()

plt.show()
fname = 'Exp1_Barplot_Reversed.pdf'
figure.savefig(fname, format='pdf')


# In[327]:


mean_FR = [np.mean(mean_unbias_FRtoFR_Happy)*100, np.mean(mean_unbias_FRtoFR_Sad)*100, np.mean(mean_unbias_FRtoFR_Afraid)*100]
mean_JP = [np.mean(mean_unbias_FRtoJP_Happy)*100, np.mean(mean_unbias_FRtoJP_Sad)*100, np.mean(mean_unbias_FRtoJP_Afraid)*100]

ci_FR_Happy = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoFR_Happy*100)-1, loc=np.mean(mean_unbias_FRtoFR_Happy*100), scale=st.sem(mean_unbias_FRtoFR_Happy*100))
ci_FR_Sad = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoFR_Sad*100)-1, loc=np.mean(mean_unbias_FRtoFR_Sad*100), scale=st.sem(mean_unbias_FRtoFR_Sad*100))
ci_FR_Afraid = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoFR_Afraid*100)-1, loc=np.mean(mean_unbias_FRtoFR_Afraid*100), scale=st.sem(mean_unbias_FRtoFR_Afraid*100))
ci_JP_Happy = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoJP_Happy*100)-1, loc=np.mean(mean_unbias_FRtoJP_Happy*100), scale=st.sem(mean_unbias_FRtoJP_Happy*100))
ci_JP_Sad = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoJP_Sad*100)-1, loc=np.mean(mean_unbias_FRtoJP_Sad*100), scale=st.sem(mean_unbias_FRtoJP_Sad*100))
ci_JP_Afraid = st.t.interval(alpha=0.95, df=len(mean_unbias_FRtoJP_Afraid*100)-1, loc=np.mean(mean_unbias_FRtoJP_Afraid*100), scale=st.sem(mean_unbias_FRtoJP_Afraid*100))


xpos_FR = np.array([0, 0.5, 1.0])
xpos_JP = np.array([0.05, 0.55, 1.05])
labels = ['Happy', 'Sad', 'Afraid']

figure, ax = plt.subplots(figsize=[6,4])
line1 = ax.plot(xpos_FR, mean_FR, color='black', marker='o', label='FR')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FR_Happy, color='black')
ax.plot([xpos_FR[1],xpos_FR[1]], ci_FR_Sad, color='black')
ax.plot([xpos_FR[2],xpos_FR[2]], ci_FR_Afraid, color='black')


line2 = ax.plot(xpos_JP, mean_JP, color='black', marker='o', label='JP', linestyle='dotted')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_JP_Happy, color='black')
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JP_Sad, color='black')
ax.plot([xpos_JP[2],xpos_JP[2]], ci_JP_Afraid, color='black')


plt.ylabel('Unbiased hit rate (%)', fontname='Arial')
plt.xticks(xpos_FR, labels, fontname='Arial')
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial')
plt.legend(handles=[line1[0], line2[0]], markerscale=0,title='Stimuli', prop={'family':'Arial'})
plt.tight_layout()

plt.show()
fname = 'Exp1_Plot_Emotion_FR.pdf'
figure.savefig(fname, format='pdf')


# In[328]:


mean_FR = [np.mean(mean_unbias_JPtoFR_Happy)*100, np.mean(mean_unbias_JPtoFR_Sad)*100, np.mean(mean_unbias_JPtoFR_Afraid)*100]
mean_JP = [np.mean(mean_unbias_JPtoJP_Happy)*100, np.mean(mean_unbias_JPtoJP_Sad)*100, np.mean(mean_unbias_JPtoJP_Afraid)*100]

ci_FR_Happy = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoFR_Happy*100)-1, loc=np.mean(mean_unbias_JPtoFR_Happy*100), scale=st.sem(mean_unbias_JPtoFR_Happy*100))
ci_FR_Sad = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoFR_Sad*100)-1, loc=np.mean(mean_unbias_JPtoFR_Sad*100), scale=st.sem(mean_unbias_JPtoFR_Sad*100))
ci_FR_Afraid = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoFR_Afraid*100)-1, loc=np.mean(mean_unbias_JPtoFR_Afraid*100), scale=st.sem(mean_unbias_JPtoFR_Afraid*100))
ci_JP_Happy = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoJP_Happy*100)-1, loc=np.mean(mean_unbias_JPtoJP_Happy*100), scale=st.sem(mean_unbias_JPtoJP_Happy*100))
ci_JP_Sad = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoJP_Sad*100)-1, loc=np.mean(mean_unbias_JPtoJP_Sad*100), scale=st.sem(mean_unbias_JPtoJP_Sad*100))
ci_JP_Afraid = st.t.interval(alpha=0.95, df=len(mean_unbias_JPtoJP_Afraid*100)-1, loc=np.mean(mean_unbias_JPtoJP_Afraid*100), scale=st.sem(mean_unbias_JPtoJP_Afraid*100))


xpos_FR = np.array([0, 0.5, 1.0])
xpos_JP = np.array([0.05, 0.55, 1.05])
labels = ['Happy', 'Sad', 'Afraid']

figure, ax = plt.subplots(figsize=[6,4])
line1 = ax.plot(xpos_FR, mean_FR, color='black', marker='o', label='FR')
ax.plot([xpos_FR[0],xpos_FR[0]], ci_FR_Happy, color='black')
ax.plot([xpos_FR[1],xpos_FR[1]], ci_FR_Sad, color='black')
ax.plot([xpos_FR[2],xpos_FR[2]], ci_FR_Afraid, color='black')


line2 = ax.plot(xpos_JP, mean_JP, color='black', marker='o', label='JP', linestyle='dotted')
ax.plot([xpos_JP[0],xpos_JP[0]], ci_JP_Happy, color='black')
ax.plot([xpos_JP[1],xpos_JP[1]], ci_JP_Sad, color='black')
ax.plot([xpos_JP[2],xpos_JP[2]], ci_JP_Afraid, color='black')


plt.ylabel('Unbiased hit rate (%)', fontname='Arial')
plt.xticks(xpos_FR, labels, fontname='Arial')
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial')
plt.legend(handles=[line1[0], line2[0]], markerscale=0,title='Stimuli', prop={'family':'Arial'})
plt.tight_layout()

plt.show()
fname = 'Exp1_Plot_Emotion_JP.pdf'
figure.savefig(fname, format='pdf')


# In[ ]:





# In[55]:


acc_toFR_N = np.concatenate([mean_unbias_FRtoFR_N, mean_unbias_JPtoFR_N])
acc_toFR_J = np.concatenate([mean_unbias_FRtoFR_J, mean_unbias_JPtoFR_J])
acc_toFR_S = np.concatenate([mean_unbias_FRtoFR_S, mean_unbias_JPtoFR_S])
acc_toFR_R = np.concatenate([mean_unbias_FRtoFR_R, mean_unbias_JPtoFR_R])

acc_toJP_N = np.concatenate([mean_unbias_FRtoJP_N, mean_unbias_JPtoJP_N])
acc_toJP_J = np.concatenate([mean_unbias_FRtoJP_J, mean_unbias_JPtoJP_J])
acc_toJP_S = np.concatenate([mean_unbias_FRtoJP_S, mean_unbias_JPtoJP_S])
acc_toJP_R = np.concatenate([mean_unbias_FRtoJP_R, mean_unbias_JPtoJP_R])

sub_type = np.concatenate([np.zeros(20), np.ones(21)])

df = pd.DataFrame(
    data =  {'sub_type':sub_type, 'acc_toFR_N': acc_toFR_N, 'acc_toFR_J': acc_toFR_J, 'acc_toFR_S': acc_toFR_S, 'acc_toFR_R':acc_toFR_R,
            'acc_toJP_N':acc_toJP_N, 'acc_toJP_J':acc_toJP_J, 'acc_toJP_S':acc_toJP_S, 'acc_toJP_R':acc_toJP_R})

df.to_csv('Exp1_data_3wayANOVA.csv',  index=False)


# In[70]:


mean_unbias_FRtoFR = np.mean(np.concatenate([mean_unbias_FRtoFR_N, mean_unbias_FRtoFR_J, mean_unbias_FRtoFR_S, mean_unbias_FRtoFR_R]))
mean_unbias_FRtoJP = np.mean(np.concatenate([mean_unbias_FRtoJP_N, mean_unbias_FRtoJP_J, mean_unbias_FRtoJP_S, mean_unbias_FRtoJP_R]))
mean_unbias_JPtoFR = np.mean(np.concatenate([mean_unbias_JPtoFR_N, mean_unbias_JPtoFR_J, mean_unbias_JPtoFR_S, mean_unbias_JPtoFR_R]))
mean_unbias_JPtoJP = np.mean(np.concatenate([mean_unbias_JPtoJP_N, mean_unbias_JPtoJP_J, mean_unbias_JPtoJP_S, mean_unbias_JPtoJP_R]))

print('Mean diff FR sub => FR vs JP: ' +  str((mean_unbias_FRtoFR - mean_unbias_FRtoJP) * 100))
print('Mean diff JP sub => JP vs FR: ' +  str((mean_unbias_JPtoJP - mean_unbias_JPtoFR) * 100))


# In[82]:


diff_Happy_FR = 100*(np.mean(mean_unbias_FRtoFR_Happy) - np.mean(mean_unbias_FRtoJP_Happy))
diff_Happy_JP = 100*(np.mean(mean_unbias_JPtoJP_Happy) - np.mean(mean_unbias_JPtoFR_Happy))

print('LFE in FR sub for Happy: ' + str(diff_Happy_FR))
print('LFE in JP sub for Happy: ' + str(diff_Happy_JP))
print('LFE for Happy, mean: ' + str(np.mean([diff_Happy_FR,diff_Happy_JP])))


# In[83]:


diff_Sad_FR = 100*(np.mean(mean_unbias_FRtoFR_Sad) - np.mean(mean_unbias_FRtoJP_Sad))
diff_Sad_JP = 100*(np.mean(mean_unbias_JPtoJP_Sad) - np.mean(mean_unbias_JPtoFR_Sad))

print('LFE in FR sub for Sad: ' + str(diff_Sad_FR))
print('LFE in JP sub for Sad: ' + str(diff_Sad_JP))
print('LFE for Sad, mean: ' + str(np.mean([diff_Sad_FR,diff_Sad_JP])))


# In[84]:


diff_Afraid_FR = 100*(np.mean(mean_unbias_FRtoFR_Afraid) - np.mean(mean_unbias_FRtoJP_Afraid))
diff_Afraid_JP = 100*(np.mean(mean_unbias_JPtoJP_Afraid) - np.mean(mean_unbias_JPtoFR_Afraid))

print('LFE in FR sub for Afraid: ' + str(diff_Afraid_FR))
print('LFE in JP sub for Afraid: ' + str(diff_Afraid_JP))
print('LFE for Afraid, mean: ' + str(np.mean([diff_Afraid_FR,diff_Afraid_JP])))


# In[ ]:




