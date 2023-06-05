This readme.txt file was generated on 2021-01-13 by Andrew I. Wilterson


GENERAL INFORMATION

1. Title of Dataset: "The Attention Schema Theory in an Artificial Neural Network Agent: Controlling Visuospatial Attention Using a Descriptive Model of Attention"

2. Author Information
	A. Principal Investigator Contact Information
		Name: Michael Graziano
		Institution: Princeton University
		Address: Princeton Neuroscience Institute
		Email: graziano@princeton.edu

	B. Associate or Co-investigator Contact Information
		Name: Andrew Wilterson
		Institution: Princeton University (at time of work)
		Address: -
		Email: Andrew.Wilterson@gmail.com
 

3. Date of data collection (single date, range, approximate date): 2020-07

4. Geographic location of data collection: Princeton, New Jersey, United States of America

5. Information about funding sources that supported the collection of the data: No outside funding sources.


SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data: N/A

2. Links to publications that cite or use the data: TBD

3. Links to other publicly accessible locations of the data: N/A

4. Links/relationships to ancillary data sets: N/A

5. Was data derived from another source? no
	A. If yes, list source(s): 

6. Recommended citation for this dataset: Wilterson, A.I., Graziano, M.S.A. (2021). The Attention Schema Theory in an Artificial Neural Network Agent: Controlling Visuospatial Attention Using a Descriptive Model of Attention [Data set]. Princeton University. https://doi.org/10.34770/8bd1-1x14


DATA & FILE OVERVIEW

1. File List: 
Experiment 1 Results.xlsx: Spreadsheet containing experimental results, in Excel format to preserve figures
Experiment 1 Results.csv: Spreadsheet containing experimental results, in csv format for open access
Experiment 2 Results.xlsx
Experiment 2 Results.csv
Experiment 3 Results.xlsx
Experiment 3 Results.csv
Experiment 1.ipynb: Jupyter Notebook containing model code and code to run the experiment
Experiment 2.ipynb
Experiment 3.ipynb

2. Relationship between files, if important: N/A

3. Additional related data collected that was not included in the current data package: N/A

4. Are there multiple versions of the dataset? no


METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
These models were inspired by recent experiment work on the Attention Schema Theory of Consciousness, see: 
Wilterson, A. I., Kemper, C., Kim, N., Webb, T. W., Reblando, A. M. W., & Graziano, M. S. A. (2020). Attention Control and the Attention Schema Theory of Consciousness. In Progress in Neurobiology

In these models we trained an artificial agent to play a simple game that required control of a simple attentional spotlight for success. In some of the models, the agent was given secodnary information regarding the location of th eattention spotlight (an attention schema).

All code is written in python and all libraries are freely available. The core of the model is built using tf-agents, which is, in turn, built on Tensorflow.

2. Methods for processing the data: 
Data were processed primarily in excel. Learning curves fit using Python.

3. Instrument- or software-specific information needed to interpret the data: 
JupyterLabs Version 2
Tensorflow 2
tf-agents
pandas
matplotlib
numpy
seaborn

4. Standards and calibration information, if appropriate: N/A

5. Environmental/experimental conditions:

Experiment 1: Trains on full model with simultaneous test on model lacking attention
	Attention-On: Testing phase wherin the agent is given the same input as in training.
	Attention-Off: Testing phase wherein the agent's attention spotlight is non-functional. The spotlight can still be moved, but does not remove visual noise.

Experiment 2: Trains on full model with simultaneous test on model lacking attention schema
	Schema-On: Testing phase wherin the agent is given the same input as in training.
	Schema-Off: Testing phase wherin the agent is not given secondary information about the location of the attentional spotlight.

Experiment 3: Trains on model lacking attention schema with simultaneous test on model lacking attention
	Attention-On: Testing phase wherin the agent is given the same input as in training.
	Attention-Off: Testing phase wherein the agent's attention spotlight is non-functional. The spotlight can still be moved, but does not remove visual noise.

6. Describe any quality-assurance procedures performed on the data: 
Each model was run 5 times with each set of result being treated as an individual subject in group analysis.


7. People involved with sample collection, processing, analysis and/or submission: 
All model code was written or adapted by Andrew Wilterson. All authors contributed to analysis.


DATA-SPECIFIC INFORMATION FOR: Expertiment <#> Results.csv

1. Number of variables: 5

2. Number of cases/rows: 1500

3. Variable List: 
Attention Score: Gives the average attention score for each iteration of training. The score is from a 50 game test phase that takes place after each training iteration. Scores are given seperately for each agent. A 10 iteration moving average is also calculated, along with a grand mean across agents. Units are in arbitrary points, wherein values greater than 0 indicate that the agent is performing its task correctly move than half the time. The attention score is determined by whether the agent's attentional spotlight overlaps the ball on any given timestep of the game.

Catch Score: Same as above, except points are derived from the agent catching or failing to catch the ball in any given game.


4. Missing data codes: 
N/A

5. Specialized formats or other abbreviations used: 
N/A
