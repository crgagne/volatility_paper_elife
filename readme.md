
## Overview

Welcome! This repository contains the data and code needed to reproduce the main figures from our recently submitted paper to Elife. The paper is titled "Impaired adaptation of learning to contingency volatility in internalizing psychopathology" and is authored by Christopher Gagne, Ondrej Zika, Peter Dayan & Sonia J. Bishop.

This Github repository is linked to and will reside permanently in an OpenScience framework repository: LINK TO BE ADDED.

## Installations

In order to fit the bifactor and behavioral models to data and to recreate the figures, you will need to install two separate conda environments, one for R and the other for Python. These environments can be installed using the following two sets of commands:

### R environment

Bash commands:
`conda create --name env_r --file requirements_r.txt`
`conda activate env_r`

### Python environment

Bash commands:
`conda create --name pymc3_3.6`
`conda activate pymc3_3.6`
`conda install pymc3=3.6`
`conda install jupyter`
`conda install seaborn`
`conda install statsmodels`

Note that most jupyter notebooks should be run using this Python environment. The only exception is the bifactor modeling notebook.


## Bifactor modeling

Participants' item-level response data, which is fit by the bifactor model, can be found in the folder: "data/item_level_data_for_bifactor_analysis". The files with the prefix 'Q_items' contain item-level response data, while the files with the prefix 'Q_labels' contain the associated questions and questionnaire labels. Separate files correspond to a different groups of subjects: experiment 1, the confirmatory analysis, or experiment 2 (see the paper for more details on these separate groups).

The code used to fit the bifactor model can be found in the folder "fitting_bifactor_model", and is contained in a Jupyer notebook with an R kernel. In order to run this code, you will need to install the R conda environment using the commands above.  

The resulting fits for the bifactor model (factor loadings and scores) are already contained in the folder "fitting_bifactor_model". These files can be used to re-create Figure 1 and its figure supplements. The code for creating these figures can be found in the folder 'figure_code', and within the Jupyter notebook titled "Figure_1.ipynb". Note that this Jupyter notebook requires a Python kernel (not an R kernel)(see second environment above). This code generates figures which can be found in the folder "figures".

## Modeling of Task Performance (Behavioral Models)

Data related to participants' task performance can be found in the folders: "data/data_raw_exp1", "data/data_raw_exp1_additional_sample", "data/raw_exp2". These folders contain trial-level data to which the behavioral models are fit. Summary data files are also contained in the "data" folder with the prefix "participant_table". These files contain participant IDs, as well as scale and subscale scores for internalizing symptoms.

The code used to load these data files and prepare them for either model fitting or figure creation can be found in the folder "data_processing_code". Here you'll also find a Jupyter notebook with an example of how to load the data from experiment 1 or experiment 2.

Fitting a behavioral model to participants data can be done using code found in the "fitting_behavioral_model_exp1" or "fitting_behavioral_model_exp2" folders. For example, the main model (see paper for more details) can be fit to behavior by running the following python command:

`python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(all)+smag+ckernel(rewpain)" --steps 2000 --steps_tune 100 --covariate Bi3itemCDM`

This code needs to be run in the Python conda environment. It also relies on additional code found in the "model_code" folder.

The code used to generate the figures related to the behavior model (Figures 3-5, and their figure supplements) can also be found in the folder "figure_code" in the Jupyter notebooks "Figures_3-4_Behavioral_Model_Exp1_Results.ipynb" and "Figure_5_Behavioral_Model_Exp2_Results.ipynb".
This code requires the model fits contained in the "fittings_behavioral_model_exp1[2]/model_fits/" folders. Due to space constraints on github, these files will be separately uploaded to the OpenScience framework repository linked to this github repository.
