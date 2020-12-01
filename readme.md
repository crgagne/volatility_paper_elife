
## Overview

Welcome! This repository contains the data and code needed to reproduce the main figures from our recently submitted paper to Elife. The paper is titled "Impaired adaptation of learning to contingency volatility in internalizing psychopathology" and is authored by Christopher Gagne, Ondrej Zika, Peter Dayan & Sonia J. Bishop.

This Github repository will also be stored within an Open Science Framework (OSF) repository, where it will be hosted permanently: https://osf.io/8mzuj/

## Steps for Reproducing the Main Analyses

1. Download the Github repository from the OSF repository
2. Download additional model fit files from the OSF repository
3. Install the required Python and R environments
4. Fit the Bifactor model to data (optional)
5. Plot the Bifactor results
6. Fit the main behavioral model to task performance data (optional)
7. Plot the main behavioral model results

## (1) Download Github Repository

You should download the entire Github repository from the OSF repository (rather than downloading individual folders). There are some code dependencies between folders.

## (2) Download Additional Files

Due to Github's space constraints, some of the large model fit files cannot be stored in the Github repository. These files can be found in a separate folder called `model_fits` in the OSF Repository. These files should be moved into Github folder `/fittings_behavioral_model/model_fits/`, once the Github folder has been downloaded to your computer.

Note that step 6 will re-create these model fits. However, we provided them in case the user wants to skip step 6 and go straight to step 7, plotting the results.

## (3) Installations

You will need to install two separate conda environments, one for R and the other for Python. These environments can be installed using the following two sets of bash commands:

#### R environment

Bash commands:
```
conda create --name env_r --file requirements_r.txt
conda activate env_r
```

#### Python environment

Bash commands:
```
conda create --name pymc3_3.6
conda activate pymc3_3.6
conda install pymc3=3.6
conda install jupyter
conda install seaborn
conda install statsmodels
```

Note that most Jupyter notebooks should be run using this Python environment. The only exception is the Bifactor modeling notebook.

To activate the environments use `conda activate pymc3_3.6` or `conda activate env_r`. To switch between environments, use `conda deactivate`.

## (4) Fitting the Bifactor to Internalizing Questionnaire Data (Optional)

Participants' item-level response data, which is fit by the bifactor model, can be found in the folder: `/data/item_level_data_for_bifactor_analysis`. The files with the prefix "Q_items" contain item-level responses, and the files with the prefix "Q_labels" contain the associated questions and questionnaire labels. Separate files correspond to a different groups of subjects: experiment 1, the confirmatory analysis, or experiment 2 (see the paper for more details on these separate groups).

The code used to fit the bifactor model can be found in the folder `fitting_bifactor_model`, and is contained in the Jupyer notebook called `Bifactor_Fitting.ipynb`. In order to run this code, you will need to install the R conda environment using the commands above and use an R kernel for the Jupyter notebook.

Note that this step is optional. The fits (factor loadings and scores) from the Bifactor model are already contained within the `fitting_bifactor_model` folder.

## (5) Plotting the Bifactor Results (Figure 1 and figure supplements)

The resulting fits for the Bifactor model can be used to re-create Figure 1 and its figure supplements. The code for creating these figures can be found in the folder `/figure_code`, and is contained in the Jupyter notebook called `Figure_1.ipynb`. This Jupyter notebook requires a Python kernel (not an R kernel) and therefore needs to be run from the Python environment (see second environment above).

This code will save Figure 1 and its figure supplements to the folder `/figures`.

## (6) Fitting Behavioral Models to Task Performance Data

Data related to participants' task performance can be found in the folders: `/data/data_raw_exp1` and `/data/data_raw_exp2`. These folders contain trial-level data, to which the behavioral models are fit.

Summary data files are also contained in the "data" folder with the prefix "participant_table". These files contain participant IDs (labeled "MID"), as well as scale and subscale scores for internalizing symptom questionnaires.

The code used to load these data files and prepare them for either model fitting or figure creation can be found in the folder `/data_processing_code`. However, this is supporting code and does not need to be called directly.

Fitting behavioral models to participants data can be done using code found in the `/fitting_behavioral_model` folder.

The main model (\#11) can be fit to experiment 1 data by running the following bash command:

```
python fit_model_to_dataset.py --modelname "11" --exp 1 --steps 2000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
```

The main model (\#11) can be fit to experiment 2 data by running the following bash command:

```
python fit_model_to_dataset.py --modelname "11" --exp 2 --steps 2000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
```

Alternate models can be fit by replacing "11" above with a different model number. See the paper for a description of the different models, numbered #1-13. For example, model \#6 can be fit to experiment 1 data using the following command.

```
python fit_model_to_dataset.py --modelname "6" --exp 1 --steps 2000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
```

The triple interaction version of the main model (\#11) can be fit by replacing '11' with '11trip'.

Note that this code needs to be run in the Python conda environment. It depends on additional supporting code found in the `/model_code` folder.

## (7) Plotting the Behavioral Model Fits (Figures 3-5 and figure supplements)

The code used to generate Figures 3-5 and their figure supplements can also be found in the folder `/figure_code`. The code is contained in the Jupyter notebooks `Figures_3-4_Behavioral_Model_Exp1_Results.ipynb` and `Figure_5_Behavioral_Model_Exp2_Results.ipynb`.

This code will expect to load the model fits from the `/fittings_behavioral_model/model_fits/` folders. These fits need to be downloaded separately and moved to this folder (see step 2).
