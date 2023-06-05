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

# Define utility functions -----------------------------------------------------

#' Compute confidence intervals corrected for the Hochberg step-up procedure
#'
#' `GetCorrectedConfInt` calculates confidence intervals for relative effect
#' estimates corrected for multiple comparisons. The correction method
#' corresponds with the Hochberg step-up procedure for correcting p-values (see
#' Efird & Nielsen, 2008).
#'
#' @param coef A vector of relative effect estimates (e.g., odds ratios)
#' @param corr_p A vector of corrected p-values, adjusted with the Hochberg
#'   procedure
#' @param alpha The significance level
#' @return A dataframe with the corrected confidence intervals. The
#'   `CI_95_Lower` column contains the corrected lower bounds, and the
#'   `CI_95_Upper` column contains the corrected upper bounds.
GetCorrectedConfInt <- function(coef, corr_p, alpha = 0.05) {
  z <- qnorm(1 - alpha / 2)
  corr_se <- coef / qnorm(1 - corr_p / 2)
  return(
    data.frame(CI_95_Lower = coef - z * corr_se,
               CI_95_Upper = coef + z * corr_se)
  )
}