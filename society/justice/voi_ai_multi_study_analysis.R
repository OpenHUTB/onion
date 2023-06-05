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

suppressPackageStartupMessages(library(emmeans))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyverse))

# Read and process data --------------------------------------------------------

# This script requires the corrected results from all studies, and so can only
# be run after running the analysis scripts for all five studies.

# Read study data.
raw_study_1_data <- read.csv('voi_ai_study_1_data.csv')
raw_study_2_data <- read.csv('voi_ai_study_2_data.csv')
raw_study_3_data <- read.csv('voi_ai_study_3_data.csv')
raw_study_4_data <- read.csv('voi_ai_study_4_data.csv')
raw_study_5_data <- read.csv('voi_ai_study_5_data.csv')

# Create combined study dataset.
overall_study_df <- (
  plyr::rbind.fill(
    raw_study_1_data, raw_study_2_data, raw_study_3_data, raw_study_4_data,
    raw_study_5_data)) %>%
  mutate(
    Study = as.factor(Study),
    Condition = as.factor(Condition),
    Passed_Comp_Test = as.logical(Passed_Comp_Test),
    Fairness_Reasoning = as.integer(Fairness_Reasoning),
  )

# Read results data.
study_1_results_df <- read.csv('voi_ai_study_1_results.csv')
study_2_results_df <- read.csv('voi_ai_study_2_results.csv')
study_3_results_df <- read.csv('voi_ai_study_3_results.csv')
study_4_results_df <- read.csv('voi_ai_study_4_results.csv')
study_5_results_df <- read.csv('voi_ai_study_5_results.csv')

# Create combined results dataset.
overall_results_df <- (
  plyr::rbind.fill(
    study_1_results_df, study_2_results_df, study_3_results_df,
    study_4_results_df, study_5_results_df)) %>%
  rename(Analysis = X) %>%
  select(Study, Analysis, Log_Odds, Adjusted_CI_95_Lower, Adjusted_CI_95_Upper,
         Adjusted_P_Value)

# Analyze principle choice -----------------------------------------------------

# Branch dataframe for visualizing principle-choice effect across all studies.
principle_choice_df <- overall_results_df %>%
  filter(Analysis == 'Principle_Choice')

# Visualize.
ggplot(data = principle_choice_df) +
  aes(x = Study, y = Log_Odds) +
  geom_hline(yintercept = 0, color = 'darkgray', linetype = 'dashed') +
  geom_linerange(aes(ymin = Adjusted_CI_95_Lower,
                     ymax = Adjusted_CI_95_Upper)) +
  geom_point(size = 2) +
  coord_flip() +
  labs(x = 'Study',
       y = 'Log odds of prioritarian\nchoice behind the VoI') +
  theme_bw()

# Analyze fairness-based reasoning ---------------------------------------------

# Branch dataframe for visualizing fairness-based reasoning effect across all
# studies.
fairness_reasoning_df <- overall_results_df %>%
  filter(Analysis == 'Fairness_Reasoning')

# Visualize.
ggplot(data = fairness_reasoning_df) +
  aes(x = Study, y = Log_Odds) +
  geom_hline(yintercept = 0, color = 'darkgray', linetype = 'dashed') +
  geom_linerange(aes(ymin = Adjusted_CI_95_Lower,
                     ymax = Adjusted_CI_95_Upper)) +
  geom_point(size = 2) +
  coord_flip() +
  labs(x = 'Study',
       y = 'Log odds of fairness-based\nreasoning behind the VoI') +
  theme_bw()

# Fit overall model to compare basic protocol and "bots" protocol.
overall_fairness.glm <- glm(
  Fairness_Reasoning ~ Study * Condition,
  data = overall_study_df,
  family = 'binomial'
)

# Estimate marginal means for all studies.
overall_fairness.emm <- emmeans(overall_fairness.glm, ~ Study : Condition)

# Create contrast list for bots vs. baseline.
interact_levels <- levels(
  interaction(overall_study_df$Study, overall_study_df$Condition))
baseline_bots_contrast <- list('baseline V - bots V' = (
  (interact_levels == '2.V') - (interact_levels == '4.V')))

# Compute difference between basic protocol and "bots" protocol.
baseline_bots_fairness.contrast <- emmeans::contrast(
  overall_fairness.emm,
  method = baseline_bots_contrast,
  type = 'response')
confint(baseline_bots_fairness.contrast)
cat('p =', summary(baseline_bots_fairness.contrast)$p.value)
