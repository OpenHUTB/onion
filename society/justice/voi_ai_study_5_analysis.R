# Copyright 2022 DeepMind Technologies Limited.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Load packages ----------------------------------------------------------------

suppressPackageStartupMessages(library(irr))
suppressPackageStartupMessages(library(rms))
suppressPackageStartupMessages(library(tidyverse))

# Define utility functions -----------------------------------------------------

source('voi_ai_utils.R')

# Read and process data --------------------------------------------------------

# Read study data.
raw_study_data <- read.csv('voi_ai_study_5_data.csv')

# Prepare data for analysis.
study_df <- raw_study_data %>%
  # Cast variables to correct types.
  mutate(
    Participant = as.factor(Participant),
    Condition = as.factor(Condition),
    Passed_Comp_Test = as.logical(Passed_Comp_Test),
    Position = as.factor(Position),
    Chose_Prioritarian = as.integer(Chose_Prioritarian),
    Repeat = as.integer(Repeat),
    Motiv_To_Change = as.logical(Motiv_To_Change),
    Motiv_To_Maximize = as.logical(Motiv_To_Maximize),
    Motiv_To_Prioritize = as.logical(Motiv_To_Prioritize),
    Risk = as.integer(Risk),
    Pol_A = as.integer(Pol_A),
    Pol_B = as.integer(Pol_B),
    Fairness_Reasoning = as.integer(Fairness_Reasoning),
  ) %>%
  # Filter out participants who failed the comprehension test.
  filter(Passed_Comp_Test)

# Create empty dataframe for pre-specified effects.
results_df <- setNames(
  data.frame(matrix(ncol = 4, nrow = 0)),
  c('Log_Odds', 'CI_95_Lower', 'CI_95_Upper', 'P_Value'))

# Analyze principle choice -----------------------------------------------------

# Fit model predicting choice of prioritarian principle.
principle_choice.glm <- glm(
  Chose_Prioritarian ~ Condition + Condition : Position,
  data = study_df,
  family = 'binomial'
)
summary(principle_choice.glm)

# Extract log odds and confidence intervals.
principle_choice_eff_df <- as.data.frame(principle_choice.glm$coefficients) %>%
  cbind(.,
        suppressMessages(confint(principle_choice.glm)),
        coef(summary(principle_choice.glm))[,4]) %>%
  setNames(., c('Log_Odds', 'CI_95_Lower', 'CI_95_Upper', 'P_Value'))

# Compute odds ratios.
principle_choice_odds_ratio_df <- principle_choice_eff_df %>%
  mutate(Log_Odds = exp(Log_Odds),
         CI_95_Lower = exp(CI_95_Lower),
         CI_95_Upper = exp(CI_95_Upper)) %>%
  rename(Odds_Ratio = Log_Odds)
principle_choice_odds_ratio_df

# Extract effect of condition on principle choice.
results_df['Principle_Choice',] <- principle_choice_eff_df['ConditionV',]

# Analyze reflective endorsement -----------------------------------------------

# Fit model predicting reflective endorsement.
repeat_choice.glm <- glm(
  Repeat ~ Condition * Motiv_To_Change,
  data = study_df,
  family = 'binomial'
)
summary(repeat_choice.glm)

# Extract log odds and confidence intervals.
repeat_choice_eff_df <- as.data.frame(repeat_choice.glm$coefficients) %>%
  cbind(.,
        suppressMessages(confint(repeat_choice.glm)),
        coef(summary(repeat_choice.glm))[,4]) %>%
  setNames(., c('Log_Odds', 'CI_95_Lower', 'CI_95_Upper', 'P_Value'))

# Compute odds ratios.
repeat_choice_odds_ratio_df <- repeat_choice_eff_df %>%
  mutate(Log_Odds = exp(Log_Odds),
         CI_95_Lower = exp(CI_95_Lower),
         CI_95_Upper = exp(CI_95_Upper)) %>%
  rename(Odds_Ratio = Log_Odds)
repeat_choice_odds_ratio_df

# Extract effect of condition and motivation to change on reflective
# endorsement.
results_df['Repeat_Choice',] <- (
  repeat_choice_eff_df['ConditionV:Motiv_To_ChangeTRUE',])

# Analyze mechanisms behind the Veil of Ignorance ------------------------------

# Branch dataframe for mechanism analysis.
veil_df <- study_df[study_df$Condition == 'V',]

# Estimate variance in choice explained by risk preferences.
mech_choice_risk.glm <- glm(
  Chose_Prioritarian ~ Risk,
  data = veil_df,
  family = 'binomial'
)

# Also fit using `rms` library to compute R-squared.
mech_choice_risk.lrm <- lrm(
  Chose_Prioritarian ~ Risk,
  data = veil_df,
)

# Extract odds-ratio estimate, p-value, and R-squared.
mech_choice_risk_df <- (
  as.data.frame(mech_choice_risk.glm$coefficients)) %>%
  exp() %>%
  cbind(.,
        coef(summary(mech_choice_risk.glm))[,4],
        mech_choice_risk.lrm$stats[['R2']] * 100) %>%
  setNames(., c('Odds_Ratio', 'P_Value', 'R_Squared'))
mech_choice_risk_df

# Estimate variance in choice explained by liberal-conservative orientation.
mech_choice_pol_a.glm <- glm(
  Chose_Prioritarian ~ Pol_A,
  data = veil_df,
  family = 'binomial'
)

# Fit using `rms` library to compute R-squared.
mech_choice_pol_a.lrm <- lrm(
  Chose_Prioritarian ~ Pol_A,
  data = veil_df,
)

# Extract odds-ratio estimate, p-value, and R-squared.
mech_choice_pol_a_df <- (
  as.data.frame(mech_choice_pol_a.glm$coefficients)) %>%
  exp() %>%
  cbind(.,
        coef(summary(mech_choice_pol_a.glm))[,4],
        mech_choice_pol_a.lrm$stats[['R2']] * 100) %>%
  setNames(., c('Odds_Ratio', 'P_Value', 'R_Squared'))
mech_choice_pol_a_df

# Estimate variance in choice explained by left-right orientation.
mech_choice_pol_b.glm <- glm(
  Chose_Prioritarian ~ Pol_B,
  data = veil_df,
  family = 'binomial'
)

# Fit using `rms` library to compute R-squared.
mech_choice_pol_b.lrm <- lrm(
  Chose_Prioritarian ~ Pol_B,
  data = veil_df,
)

# Extract odds-ratio estimate, p-value, and R-squared.
mech_choice_pol_b_df <- (
  as.data.frame(mech_choice_pol_b.glm$coefficients)) %>%
  exp() %>%
  cbind(.,
        coef(summary(mech_choice_pol_b.glm))[,4],
        mech_choice_pol_b.lrm$stats[['R2']] * 100) %>%
  setNames(., c('Odds_Ratio', 'P_Value', 'R_Squared'))
mech_choice_pol_b_df

# Estimate variance in choice explained by fairness-based reasoning.
mech_choice_fair.glm <- glm(
  Chose_Prioritarian ~ Fairness_Reasoning,
  data = veil_df,
  family = 'binomial'
)

# Fit using `rms` library to compute R-squared.
mech_choice_fair.lrm <- lrm(
  Chose_Prioritarian ~ Fairness_Reasoning,
  data = veil_df,
)

# Extract odds-ratio estimate, p-value, and R-squared.
mech_choice_fair_df <- (
  as.data.frame(mech_choice_fair.glm$coefficients)) %>%
  exp() %>%
  cbind(.,
        coef(summary(mech_choice_fair.glm))[,4],
        mech_choice_fair.lrm$stats[['R2']] * 100) %>%
  setNames(., c('Odds_Ratio', 'P_Value', 'R_Squared'))
mech_choice_fair_df

# Analyze fairness-based reasoning ---------------------------------------------

# Compute interrater agreement for fairness-based reasoning.
fairness_ratings <- t(as.matrix(
  study_df[, c('Rater_1_Fairness', 'Rater_2_Fairness')]))
kripp.alpha(fairness_ratings, method = 'nominal')

# Fit model predicting fairness-based reasoning.
fairness_reasoning.glm <- glm(
  Fairness_Reasoning ~ Condition,
  data = study_df,
  family = 'binomial'
)
summary(fairness_reasoning.glm)

# Extract log odds and confidence intervals.
fairness_reasoning_eff_df <- (
  as.data.frame(fairness_reasoning.glm$coefficients)) %>%
  cbind(.,
        suppressMessages(confint(fairness_reasoning.glm)),
        coef(summary(fairness_reasoning.glm))[,4]) %>%
  setNames(., c('Log_Odds', 'CI_95_Lower', 'CI_95_Upper', 'P_Value'))

# Compute odds ratios.
fairness_reasoning_odds_ratio_df <- fairness_reasoning_eff_df %>%
  mutate(Log_Odds = exp(Log_Odds),
         CI_95_Lower = exp(CI_95_Lower),
         CI_95_Upper = exp(CI_95_Upper)) %>%
  rename(Odds_Ratio = Log_Odds)
fairness_reasoning_odds_ratio_df

# Extract effect of condition on fairness-based reasoning.
results_df['Fairness_Reasoning',] <- fairness_reasoning_eff_df['ConditionV',]

# Adjust for multiple comparisons ----------------------------------------------

# Compute adjusted p-value according to Hochberg (1988) and adjusted confidence
# intervals according to Efird and Nielson (2008).
results_df['Adjusted_P_Value'] <- p.adjust(results_df[['P_Value']],
                                           method = 'hochberg')
results_df[c('Adjusted_CI_95_Lower', 'Adjusted_CI_95_Upper')] <- (
  GetCorrectedConfInt(results_df[['Log_Odds']],
                      results_df[['Adjusted_P_Value']]))
results_df[, c(
  'Log_Odds', 'Adjusted_CI_95_Lower', 'Adjusted_CI_95_Upper',
  'Adjusted_P_Value')]

# Save data --------------------------------------------------------------------

# Add study variable.
adjusted_results_df <- results_df %>%
  mutate(Study = 5) %>%
  select(Study, Log_Odds, Adjusted_CI_95_Lower, Adjusted_CI_95_Upper,
         Adjusted_P_Value)

# Save data.
write.csv(as.data.frame(adjusted_results_df), 'voi_ai_study_5_results.csv')