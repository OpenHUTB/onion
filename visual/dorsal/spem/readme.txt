spemconnect_analyse_behav_osf.R 

script to analyse behavioural data from SPEM-CONNECT study (Functional 
connectivity during smooth pursuit eye movements by Rebekka 
Schröder, Anna-Maria Kasparbauer, Maria Steffens, Inga Meyhöfer, Peter Trautner
and Ulrich Ettinger)
 - descriptive statistics for pursuit gain, saccade rate and RMSE
 - t-Tests to compare pursuit gain, saccade rate and RMSE between 
   the two conditions (0.2 Hz pursuit vs. 0.4 Hz pursuit)
 - correlations between pursuit gain, saccade rate, RMSE and 
   extracted ROI BOLD data from 10 seed regions (left and right LGN, V1, V5,
   PPC and FEF) for the two pursuit conditions



spemconnect_data_osf.rda 

contains anonymised data necessary to run above script

Variable description:  

 $ id        : chr  participant id
 $ sex       : int  sex, 1 = female, 2 = male
 $ age       : int  age in years
 $ sacfreq_02: num  saccade rate (N/s) at 0.2 Hz target stimulus frequency 
 $ sacfreq_04: num  saccade rate (N/s) at 0.4 Hz target stimulus frequency
 $ gain_02   : num  smooth pursuit velocity gain at 0.2 Hz target stimulus frequency
 $ gain_04   : num  smooth pursuit velocity gain at 0.4 Hz target stimulus frequency
 $ rmse_02   : num  root mean square error (in pixel) at 0.2 Hz target stimulus frequency
 $ rmse_04   : num  root mean square error (in pixel) at 0.2 Hz target stimulus frequency
 $ LGNl2     : num  mean activity in left LGN at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ LGNr2     : num  mean activity in right ft LGN at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ LGNl4     : num  mean activity in left LGN at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ LGNr4     : num  mean activity in right LGN at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ V1l2      : num  mean activity in left V1 at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ V1r2      : num  mean activity in right V1 at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ V1l4      : num  mean activity in left V1 at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ V1r4      : num  mean activity in right V1 at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ V5l2      : num  mean activity in left V5 at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ V5r2      : num  mean activity in right V5 at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ V5l4      : num  mean activity in left V5 at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ V5r4      : num  mean activity in right V5 at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ PARl2     : num  mean activity in left PPC at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ PARr2     : num  mean activity in right PPC at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ PARl4     : num  mean activity in left PPC at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ PARr4     : num  mean activity in right PPC at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ FEFl2     : num  mean activity in left FEF at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ FEFr2     : num  mean activity in right FEF at 0.2 Hz target stimulus frequency (extracted via MarsBar)
 $ FEFl4     : num  mean activity in left FEF at 0.4 Hz target stimulus frequency (extracted via MarsBar)
 $ FEFr4     : num  mean activity in right FEF at 0.4 Hz target stimulus frequency (extracted via MarsBar)


Participants N3 and N26 were not included in fMRI analyses due to failed preprocessing; therefore no activity data is available for these participants




fMRI analysis scripts:

first_level_gppi: scripts for each seed region to perform first level gPPI analysis

first_level_task: script to perform first-level task analysis

second_level_ppi_regression: scripts to perform regression analysis for correlations between gPPI contrasts 
			     and pursuit gain, RMSE and saccadic frequency, separately for each seed region (left and right)
		    	     and .2 Hz and .4 Hz

second_level_ppi_ttest: scripts to perform second-level analysis (t-test) on gPPI (created in first_level_gppi)

second_level_task_regression: scripts to perform regression analysis for correlations between task contrasts 
			      and pursuit gain, RMSE and saccadic frequency 

second_level_task_ttest: scripts for each task t-test for the task contrasts as the second level 
			(.2 Hz vs. fixation, .4 Hz vs. fixation, .2 Hz vs. .4 Hz, .4 Hz vs. .2 Hz)