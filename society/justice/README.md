# Using the Veil of Ignorance to Align AI Systems with Principles of Justice

Datasets and analysis scripts for studies conducted by Weidinger, McKee, et al.
(2023).


## Contents

This repo contains seven analysis scripts to reproduce the results and plots
presented in Weidinger, McKee, et al. (2023):

  * `voi_ai_utils.R`: Script defining utility functions.
  * `voi_ai_study_1_analysis.R`: Analysis script for study 1.
  * `voi_ai_study_2_analysis.R`: Analysis script for study 2.
  * `voi_ai_study_3_analysis.R`: Analysis script for study 3.
  * `voi_ai_study_4_analysis.R`: Analysis script for study 4.
  * `voi_ai_study_5_analysis.R`: Analysis script for study 5.
  * `voi_ai_multi_study_analysis.R`: Analysis script for multi-study results.

It also contains five datasets:

  * `voi_ai_study_1_data.csv`: Participant responses for study 1.
  * `voi_ai_study_2_data.csv`: Participant responses for study 2.
  * `voi_ai_study_3_data.csv`: Participant responses for study 3.
  * `voi_ai_study_4_data.csv`: Participant responses for study 4.
  * `voi_ai_study_5_data.csv`: Participant responses for study 5.

## Running a script

In an R session, set the working directory to the folder containing the script
and datasets associated with one of the studies. The script can then be run with
`source("[script_name].R", echo = T)`. This will automatically print each
expression in the script in order, as well as any output resulting from
evaluating the expression (e.g., the results from a statistical test).

For example, to reproduce the results and plots for study 1, run the following
command:

```
source("voi_ai_study_1_analysis.R", echo = T)
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

# 参考
原始代码和数据[链接](https://osf.io/eapqu/) 。
