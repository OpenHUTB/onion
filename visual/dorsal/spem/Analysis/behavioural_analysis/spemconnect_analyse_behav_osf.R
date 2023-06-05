# SPEM-CONNECT SCRIPT -----------------------------------------------------

# -  Script by Rebekka Schröder, rebekka.schroeder@uni-bonn.de, University of Bonn, Germany
#    to analyze behavioural data of Spem-PPI-Connectivity Study: 
#    "Functional Connectivity During Smooth Pursuit Eye Movements" by Rebekka Schröder, 
#     Anna-Maria Kasparbauer, Inga Meyhöfer, Maria Steffens, Peter Trautner and Ulrich Ettinger
  
# -  behavioural data was collected during a smooth pursuit task at two different target frequencies
#    0.2 Hz vs. 0.4 Hz
# -  data were preprocessed -> 3 variables were extracted (pursuit gain, saccade rate, root mean square
#    error) for each of the two conditions
# -  data contains  44 of 57 participants, as 13 participants could not be analyzed due to poor 
#    eyetracking data quality

# -  before running the script make sure to have the following packages installed: tidyR, dplyr, Hmisc
# -  please customize path in line 27 to the folder, where you saved the data 'spemconnect_data_osf.rda'

# load data ---------------------------------------------------------------


packages <- c("tidyr", "dplyr", "Hmisc") # append if necesssary
# lapply(packages, library, character.only = TRUE)


# prepare data ------------------------------------------------------------

# setwd("/data2/whd/workspace/sot/hart/spem/Data/")                                                      ## customize
setwd("Z:/data2/whd/workspace/sot/hart/spem/Data")
load('spemconnect_data_osf.rda')

data_long <- data_osf %>%
  gather(frequency, value, sacfreq_02:rmse_04)%>%
  extract(frequency, c("DV", "freq"), regex = "(sacfreq|gain|rmse)_(02|04)")%>%
  spread("DV", value)


# t-Tests -----------------------------------------------------------------
# calculate pairwise-t-Test for each of the three dependent variables (gain, rmse, sacfreq)

data <- data_long

results_ttest_gain    <- t.test(data$gain    ~ data$freq, paired = TRUE)
results_ttest_rmse    <- t.test(data$rmse    ~ data$freq, paired = TRUE)
results_ttest_sacfreq <- t.test(data$sacfreq ~ data$freq, paired = TRUE)


# Calculate dav according to Lakens 2013 Front Psychol --------------------
# effect sizes for the three t-Tests from the previous section

# gain
sd_freq_gain  <-  data %>%
                  group_by(freq) %>%
                  summarise(sd = sd(gain))
dav_gain      <-  results_ttest_gain$estimate / ((sd_freq_gain[1, 2] + sd_freq_gain[2, 2])/2)

# rmse
sd_freq_rmse  <-  data %>%
                  group_by(freq) %>%
                  summarise(sd = sd(rmse))
dav_rmse      <-  results_ttest_rmse$estimate / ((sd_freq_rmse[1, 2] + sd_freq_rmse[2, 2])/2)

# saccade frequency
sd_freq_sacfreq   <- data %>%
                  group_by(freq) %>%
                  summarise(sd = sd(sacfreq))
dav_sacfreq   <-  results_ttest_sacfreq$estimate / ((sd_freq_sacfreq[1, 2] + sd_freq_sacfreq[2, 2])/2)

# result summary t-Tests --------------------------------------------------

summary_ttest_gain    <- paste0("t(",results_ttest_gain$parameter,    ") = ", round(results_ttest_gain$statistic,2),    ", p = ", format(round(results_ttest_gain$p.value   ,2),    nsmall=3), ", d = ", round(dav_gain,2))
summary_ttest_rmse    <- paste0("t(",results_ttest_rmse$parameter,    ") = ", round(results_ttest_rmse$statistic,2),    ", p = ", format(round(results_ttest_rmse$p.value   ,2),    nsmall=3), ", d = ", round(dav_rmse, 2))
summary_ttest_sacfreq <- paste0("t(",results_ttest_sacfreq$parameter, ") = ", round(results_ttest_sacfreq$statistic,2), ", p = ", format(round(results_ttest_sacfreq$p.value   ,2), nsmall=3), ", d = ", round(dav_sacfreq, 2))


# descriptives gain, rmse, saccade rate -----------------------------------

data <- data_long

table_desc <- data %>%
  group_by(freq)%>%
  summarise(mean_gain    = round(mean(gain),2),
            sd_gain      = paste0("(",round(sd  (gain),2), ")"),
            mean_sacfreq = round(mean(sacfreq),2),
            sd_sacfreq   = paste0("(",round(sd(sacfreq),2), ")"),
            mean_rmse    = round(mean(rmse),2),
            sd_rmse      = paste0("(",round(sd(rmse),2), ")"))%>%
  unite(gain, mean_gain, sd_gain, sep= "")%>%
  unite(sacfreq, mean_sacfreq, sd_sacfreq, sep= "")%>%
  unite(rmse, mean_rmse, sd_rmse, sep= "")

# calculate correlations between ROI data and behavioural outcomes --------

data <- data_osf

flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  = (cormat)[ut],
    p = pmat[ut]
  )
}

names_data_ROIS <- c(
  "vp",
  "LGNl2",
  "LGNr2",
  "LGNl4",
  "LGNr4",
  "V1l2",
  "V1r2",
  "V1l4",
  "V1r4",
  "V5l2",
  "V5r2",
  "V5l4",
  "V5r4",
  "PARl2",
  "PARr2",
  "PARl4",
  "PARr4",
  "FEFl2",
  "FEFr2",
  "FEFl4",
  "FEFr4"
)

names_dvs <-
  c("sacfreq_02",
    "sacfreq_04",
    "gain_02",
    "gain_04",
    "rmse_02",
    "rmse_04")


names_data_ROIS_2 <- names_data_ROIS[endsWith(names_data_ROIS, "2")]
names_data_ROIS_4 <- names_data_ROIS[endsWith(names_data_ROIS, "4")]

names_dvs_2 <- names_dvs[endsWith(names_dvs, "2")]
names_dvs_4 <- names_dvs[endsWith(names_dvs, "4")]

x2 <- data[names_data_ROIS_2]
y2 <- data[names_dvs_2]
corr_matrix_2 <- Hmisc::rcorr(as.matrix(cbind(x2, y2)))
flat_corr_matrix_2 <- flattenCorrMatrix(corr_matrix_2$r, corr_matrix_2$P)
corr_matrix_2_new <- flat_corr_matrix_2%>%
  filter(column %in% names_dvs_2,
         ! row %in% names_dvs_2)

x4 <- data[names_data_ROIS_4]
y4 <- data[names_dvs_4]
corr_matrix_4 <- rcorr(as.matrix(cbind(x4, y4)))
flat_corr_matrix_4 <-  flattenCorrMatrix(corr_matrix_4$r, corr_matrix_4$P)

corr_matrix_4_new <- flat_corr_matrix_4%>%
  filter(column %in% names_dvs_4,
         ! row %in% names_dvs_4)

# results summary correlations --------------------------------------------

order = c("LGN", "V1", "V5", "PAR", "FEF")

corr_matrix_summary <- bind_cols(corr_matrix_2_new, corr_matrix_4_new)%>%
  extract(row, into = c("Region", "Hemisphere"), regex = "(LGN|V1|V5|PAR|FEF)(l|r)", remove = TRUE)%>%
  mutate(column = recode(column, "sacfreq_02" = "saccade rate",
                                 "gain_02" = "pursuit gain", 
                                 "rmse_02" = "RMSE"))%>%
  mutate(Region =  factor(Region, levels = order))%>%
  arrange(Region)%>%
  select(Region, column, Hemisphere, cor, p, cor1, p1)%>%
  rename("Low frequency R" = cor, "Low frequency p" = p, "High frequency R" = cor1, "High frequency p" = p1)



# SUMMARY -----------------------------------------------------------------

# - t-Test results can be found in the following variables
#    - for smooth pursuit gain:                    summary_ttest_gain   , more info in results_ttest_gain
#    - for smooth pursuit root mean square error:  summary_ttest_rmse   , more info in results_ttest_rmse
#    - for saccade rate during smooth pursuit:     summary_ttest_sacfreq, more info in results_ttest_sacfreq
#
# - descriptive statistics (mean and standard deviation[in brackets]) for gain, RMSE and saccade rate 
#   can be found in table_desc
#
# - correlation table between extracted BOLD and the three behavioural outcomes can be found in 
#   corr_matrix_summary

