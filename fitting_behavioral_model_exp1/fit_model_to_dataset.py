import sys
import sys
import os

#Pymc
import pymc3 as pm
import theano
import theano.tensor as T

import numpy as np
import pickle
import imp
import sys
import pandas as pd
import datetime


def invlogit(p):
    return 1 / (1 + np.exp(-p))

# my imports
sys.path.append('../')
sys.path.append('../model_code/')
sys.path.append('../data_processing_code/')

import get_data
from get_data import get_data

import model_base2
from model_base2 import *


import argparse

def main():
    '''
    Examples:
    python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(all)+smag+ckernel(rewpain)" --steps 2000 --steps_tune 100 --covariate Bi3itemCDM
    python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(all)+smag+ckernel(rewpain)" --steps 2000 --steps_tune 100 --covariate Bi3itemCDMsubset

    # Model Comparison
    python fit_model_to_dataset.py --seed 3 --modelname "single_ev" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "single" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(lr)" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(other)" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(all)" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(all)+triple" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(all)+smag" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(all)+smag+eps" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "double+goodbad(all)+smag" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "doubleleaky+goodbad(all)+smag" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(all)+smag+ckernel(rewpain)" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "double+goodbad(all)+smag+ckernel(rewpain)" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM
    #python fit_model_to_dataset.py --seed 3 --modelname "doubleleaky+goodbad(all)+smag+ckernel(rewpain)" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM

    # Interaction model
    python fit_model_to_dataset.py --seed 3 --modelname "single+goodbad(triple)+smag+ckernel(rewpain)" --steps 1000 --steps_tune 100 --covariate Bi3itemCDM

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--modelname', '-m', type=str, default=None)
    parser.add_argument('--steps', '-st', type=int, default=1000)
    parser.add_argument('--steps_tune', '-stt', type=int, default=100)
    parser.add_argument('--covariate', '-c', type=str, default='None')
    parser.add_argument('--hierarchical','-hh',type=str,default='True')
    parser.add_argument('--task','-tt',type=str,default='both')
    parser.add_argument('--subset','-sub',type=str,default='all')
    parser.add_argument('--covariatemask','-cm',type=str,default='None')

    args = parser.parse_args()
    print(args.modelname)
    print(args.steps)
    print(args.steps_tune)
    print(args.hierarchical)
    print(args.subset)
    print(args.covariatemask)
    args.hierarchical=eval(args.hierarchical)

    if 'single' in args.modelname:
        import model_single_est_flex as model_specific
    elif 'single_ev' in args.modelname:
        import model_single_est_EV_flex as model_specific
    elif 'double' in args.modelname and 'leaky' not in args.modelname:
        import model_double_est_flex as model_specific
    elif 'doubleleaky' in args.modelname:
        import model_double_est_leaky_beta_flex as model_specific

    if args.steps>500:
        save_state_variables=False
    else:
        save_state_variables=False

    B_max = 10
    nonlinear_indicator = 0 # mag diff scaled

    if args.task=='both':

        if args.modelname=='single_ev':
            # Model 1 #

            params=['lr_baseline','lr_stabvol',
             'Gamma_baseline','Gamma_stabvol',
             'Binv_baseline','Binv_stabvol',
             'lr_rewpain','lr_rewpain_stabvol',
            'Gamma_rewpain','Gamma_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='single':
            # Model 2 #

            params=['lr_baseline','lr_stabvol',
             'Amix_baseline','Amix_stabvol',
             'Binv_baseline','Binv_stabvol',
             'lr_rewpain','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_stabvol',
            ]
            B_max = 10


        if args.modelname=='single+goodbad(lr)':
            # Model 3 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_stabvol',
             'Binv_baseline','Binv_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='single+goodbad(other)':
            # Model 4 #

            params=['lr_baseline','lr_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'lr_rewpain','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            ]
            B_max = 10


        if args.modelname=='single+goodbad(all)':
            # Model 5 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='single+goodbad(all)+triple':
            # Model 6 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol','lr_rewpain_goodbad_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol','Amix_rewpain_goodbad_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol','Binv_rewpain_goodbad_stabvol',
            ]
            B_max = 10

        if args.modelname=='single+goodbad(all)+smag':
            # Model 7 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
            'mag_baseline',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            'mag_rewpain'
            ]
            B_max = 10

        if args.modelname=='single+goodbad(all)+smag+eps':
            # Model 8 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
            'mag_baseline',
            'eps_baseline',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            'mag_rewpain',
            'eps_rewpain'
            ]
            B_max = 10

        if args.modelname=='double+goodbad(all)+smag':
            # Model 9 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
            'mag_baseline',
            'decay_baseline','decay_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            'mag_rewpain',
            'decay_rewpain','decay_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='doubleleaky+goodbad(all)+smag':
            # Model 10 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
            'mag_baseline',
            'decay_baseline','decay_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            'mag_rewpain',
            'decay_rewpain','decay_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='single+goodbad(all)+smag+ckernel(rewpain)':
            # Model 11 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                'lr_c_baseline',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'Bc_baseline',
            'mag_baseline',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            'Bc_rewpain',
            'mag_rewpain'
            ]
            B_max = 10

        if args.modelname=='double+goodbad(all)+smag+ckernel(rewpain)':
            # Model 12 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                'lr_c_baseline',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'Bc_baseline',
            'mag_baseline',
            'decay_baseline','decay_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            'Bc_rewpain',
            'mag_rewpain',
            'decay_rewpain','decay_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='doubleleaky+goodbad(all)+smag+ckernel(rewpain)':
            # Model 13 #

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                'lr_c_baseline',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'Bc_baseline',
            'mag_baseline',
            'decay_baseline','decay_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            'Bc_rewpain',
            'mag_rewpain',
            'decay_rewpain','decay_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='single+goodbad(triple)+smag+ckernel(rewpain)':
            # Model 11 with triple interaction

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                'lr_c_baseline',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'Bc_baseline',
            'mag_baseline',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol','lr_rewpain_goodbad_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            'Bc_rewpain',
            'mag_rewpain'
            ]
            B_max = 10

    elif args.task=='rew' or args.task=='pain':

        if args.modelname=='single+goodbad(all)+smag+ckernel':

            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
                'lr_c_baseline',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'Bc_baseline',
            'mag_baseline']
            B_max = 10


    # load data
    dftmp = pd.read_csv('../data/participant_table_exp1.csv')
    data = get_data(dftmp)

    if args.covariatemask =='signifBi1':
        sigset = ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
            'Binv_baseline','Binv_rewpain_goodbad','Amix_baseline','Amix_goodbad']
        u_covariate_mask = np.array([1 if p in sigset else 0 for p in params])
        u_covariate_mask = u_covariate_mask[:,np.newaxis]
        mask_name = 'masksignifBi1_all'
    elif args.covariatemask =='signifBi1LR':
        sigset = ['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol']
        u_covariate_mask = np.array([1 if p in sigset else 0 for p in params])
        u_covariate_mask = u_covariate_mask[:,np.newaxis]
        mask_name = 'masksignifBi1'
    else:
        u_covariate_mask = None
        mask_name=''


    if args.task=='both':

        if args.subset=='all':

            # prepare for model code
            subj_indices = slice(0,157)
            Nboth = data['Nboth']

            Y = {}
            Y['participants_choice'] = data['participants_choice'][:,subj_indices]

            X = {}
            for var in ['outcomes_c_flipped','mag_1_c','mag_0_c','stabvol','rewpain']:
                X[var]=data[var][:,subj_indices]
            X['NN']=X[var].shape[1]
            X['Nboth']=data['Nboth']
            X['Nrewonly']=data['Nrewonly']
            X['Npainonly']=data['Npainonly']

            C = {}
            for stem in ['Bi1item_w_j_scaled','Bi2item_w_j_scaled','Bi3item_w_j_scaled',
                         'Bi1item_subset_scaled','Bi2item_subset_scaled','Bi3item_subset_scaled',
                         'PCA_1_scaled','PCA_2_scaled','PCA_3_scaled',
                         'Oblimin2_1_scaled','Oblimin2_2_scaled',
                         'PSWQ_scaled_residG','MASQAA_scaled_residG','MASQAD_scaled_residG',
                         'BDI_scaled_residG','STAIanx_scaled_residG','STAI_scaled_residG',
                          'PSWQ_scaled_residPC1','MASQAA_scaled_residPC1','MASQAD_scaled_residPC1',
                          'BDI_scaled_residPC1','STAIanx_scaled_residPC1','STAI_scaled_residPC1',
                         'PSWQ_scaled','MASQAA_scaled','MASQAD_scaled',
                         'BDI_scaled','STAIanx_scaled','STAI_scaled']:
                for tail in ['both','pain_only','rew_only']:
                    C[stem+'_'+tail]=data[stem+'_'+tail]


            for g in ['control_cdm','control_mfmri','MDD','GAD']:
                C['group_diag4_ind_'+g+'_both']=data['group_diag4_ind_'+g+'_both']
                C['group_diag4_ind_'+g+'_rew_only']=data['group_diag4_ind_'+g+'_rew_only']
                C['group_diag4_ind_'+g+'_pain_only']=data['group_diag4_ind_'+g+'_pain_only']


        # restrict subjects for supplemental analysis
        elif args.subset=='mix12_pred66' or args.subset=='mix12_pred75':

            if args.subset=='mix12_pred66':
                # exclude list
                exclude = ['cb105', 'cb108', 'cb12', 'cb20', 'cb28', 'cb44', 'cb55', 'cb63',
                       'cb74', 'cb75', 'cb79', 'cb80', 'cb84', 'cb89', 'cb92', 'n29',
                       'n31', 'n39', 'cb65', 'cb2', 'cb22', 'cb88']

            elif args.subset=='mix12_pred75':
                exclude = ['cb105' 'cb108' 'cb12' 'cb13' 'cb20' 'cb28' 'cb44' 'cb55' 'cb63' 'cb73'
                         'cb74' 'cb75' 'cb79' 'cb80' 'cb81' 'cb84' 'cb89' 'cb92' 'cb97' 'n29'
                         'n31' 'n39' 'cb5' 'cb65' 'cb70' 'cb77' 'cb82' 'cb2' 'cb22' 'cb88']

            # calculate some idx
            include_idx = np.array(np.array([0 if MID in exclude else 1 for MID in data["MID_combined"]])==1)
            include_idx_both = np.array(np.array([0 if MID in exclude else 1 for MID in data["MID_has_both"]])==1)
            include_idx_pain = np.array(np.array([0 if MID in exclude else 1 for MID in data["MID_pain_only"]])==1)
            include_idx_rew = np.array(np.array([0 if MID in exclude else 1 for MID in data["MID_rew_only"]])==1)

            Y = {}
            Y['participants_choice'] = data['participants_choice'][:,include_idx]

            X = {}
            for var in ['outcomes_c_flipped','mag_1_c','mag_0_c','stabvol','rewpain']:
                X[var]=data[var][:,include_idx]
            X['NN']=X[var].shape[1]
            X['Nboth']=len([MID for MID in data['MID_has_both'] if MID not in exclude])
            X['Nrewonly']=len([MID for MID in data['MID_rew_only'] if MID not in exclude])
            X['Npainonly']=len([MID for MID in data['MID_pain_only'] if MID not in exclude])

            C = {}
            C['Bi1item_w_j_scaled_both']=data['Bi1item_w_j_scaled_both'][include_idx_both]
            C['Bi2item_w_j_scaled_both']=data['Bi2item_w_j_scaled_both'][include_idx_both]
            C['Bi3item_w_j_scaled_both']=data['Bi3item_w_j_scaled_both'][include_idx_both]
            C['Bi1item_w_j_scaled_pain_only']=data['Bi1item_w_j_scaled_pain_only'][include_idx_pain]
            C['Bi2item_w_j_scaled_pain_only']=data['Bi2item_w_j_scaled_pain_only'][include_idx_pain]
            C['Bi3item_w_j_scaled_pain_only']=data['Bi3item_w_j_scaled_pain_only'][include_idx_pain]
            C['Bi1item_w_j_scaled_rew_only']=data['Bi1item_w_j_scaled_rew_only'][include_idx_rew]
            C['Bi2item_w_j_scaled_rew_only']=data['Bi2item_w_j_scaled_rew_only'][include_idx_rew]
            C['Bi3item_w_j_scaled_rew_only']=data['Bi3item_w_j_scaled_rew_only'][include_idx_rew]


            print('new NN:'+str(X['NN']))
            print('new Nboth:'+str(X['Nboth']))
            print('new Nrewonly:'+str(X['Nrewonly']))
            print('new Npainonly:'+str(X['Npainonly']))

        elif args.subset=='mike_fmri':

            ## NOT DONE ##
            # exclude list
            exclude = [MID for MID in data['MID_all_unique'] if 'c' in MID]
            exclude.append('n40') # only pain only (not all participants have both)

            # calculate some idx
            include_idx = np.array(np.array([0 if MID in exclude else 1 for MID in data["MID_combined"]])==1)
            include_idx_both = np.array(np.array([0 if MID in exclude else 1 for MID in data["MID_has_both"]])==1)
            include_idx_pain = np.array(np.array([0 if MID in exclude else 1 for MID in data["MID_pain_only"]])==1)
            include_idx_rew = np.array(np.array([0 if MID in exclude else 1 for MID in data["MID_rew_only"]])==1)

            Y = {}
            Y['participants_choice'] = data['participants_choice'][:,include_idx]

            X = {}
            for var in ['outcomes_c_flipped','mag_1_c','mag_0_c','stabvol','rewpain']:
                X[var]=data[var][:,include_idx]
            X['NN']=X[var].shape[1]
            X['Nboth']=len([MID for MID in data['MID_has_both'] if MID not in exclude])
            X['Nrewonly']=len([MID for MID in data['MID_rew_only'] if MID not in exclude])
            X['Npainonly']=len([MID for MID in data['MID_pain_only'] if MID not in exclude])

            C = {}
            C['Bi1item_w_j_scaled_both']=data['Bi1item_w_j_scaled_both'][include_idx_both]
            C['Bi2item_w_j_scaled_both']=data['Bi2item_w_j_scaled_both'][include_idx_both]
            C['Bi3item_w_j_scaled_both']=data['Bi3item_w_j_scaled_both'][include_idx_both]
            C['Bi1item_w_j_scaled_pain_only']=data['Bi1item_w_j_scaled_pain_only'][include_idx_pain]
            C['Bi2item_w_j_scaled_pain_only']=data['Bi2item_w_j_scaled_pain_only'][include_idx_pain]
            C['Bi3item_w_j_scaled_pain_only']=data['Bi3item_w_j_scaled_pain_only'][include_idx_pain]
            C['Bi1item_w_j_scaled_rew_only']=data['Bi1item_w_j_scaled_rew_only'][include_idx_rew]
            C['Bi2item_w_j_scaled_rew_only']=data['Bi2item_w_j_scaled_rew_only'][include_idx_rew]
            C['Bi3item_w_j_scaled_rew_only']=data['Bi3item_w_j_scaled_rew_only'][include_idx_rew]


            print('new NN:'+str(X['NN']))
            print('new Nboth:'+str(X['Nboth']))
            print('new Nrewonly:'+str(X['Nrewonly']))
            print('new Npainonly:'+str(X['Npainonly']))

        # Create base model (i.e. prior), embedding factors into the priors
        idx_first_reward_pain= np.min([pi for (pi,p) in enumerate(params) if 'rew' in p])
        print('compiling base model')
        model = create_model_base(X,Y,C, # Changed here
                        params=params,
                        K=len(params),
                        Konetask=idx_first_reward_pain,
                        rew_slice=slice(0,idx_first_reward_pain),
                        pain_slice=slice(0,idx_first_reward_pain),
                        split_by_reward=True,
                        includes_subjs_with_one_task=True,
                        covariate=args.covariate,
                        hierarchical=args.hierarchical,
                        covv='diag',
                        coding='deviance',
                        u_covariate_mask=u_covariate_mask)

    elif args.task=='rew' or args.task=='pain':

        if args.task =='rew':
            subj_indices = np.where((data['rewpain'][0,:]==1))[0]
        elif args.task =='pain':
            subj_indices = np.where((data['rewpain'][0,:]==-1))[0]
        print(len(subj_indices))

        Y = {}
        Y['participants_choice'] = data['participants_choice'][:,subj_indices]

        X = {}
        for var in ['outcomes_c_flipped','mag_1_c','mag_0_c','stabvol','rewpain']:
            X[var]=data[var][:,subj_indices]
        X['NN']=X[var].shape[1]
        X['Nboth']=data['Nboth']+data['N'+args.task+'only']

        C = {}
        for trait in ['Bi1item_w_j_scaled','Bi2item_w_j_scaled','Bi3item_w_j_scaled']:
            C[trait+'_both']=np.array(list(data[trait+'_both'])+list(data[trait+'_'+args.task+'_only']))

        model = create_model_base(X,Y,C,
                        params=params,
                        K=len(params),
                        Konetask=len(params),
                        rew_slice=slice(0),
                        pain_slice=slice(0),
                        split_by_reward=False,
                        includes_subjs_with_one_task=False,
                        one_task_only=True,
                        covariate=args.covariate,
                        hierarchical=args.hierarchical,
                        covv='diag',
                        coding='deviance',
                        u_covariate_mask=u_covariate_mask,
                       )

    # Create likelihood model
    print('compiling specific model')
    model = model_specific.combined_prior_model_to_choice_model(X,Y,
                                                                    param_names=params,
                                                                    model=model,
                                                                    save_state_variables=save_state_variables,
                                                                    B_max=B_max,nonlinear_indicator=nonlinear_indicator)

    # Save name
    if args.hierarchical:
        hier='hier'
    else:
        hier='nonhier'
    print('saving')
    now = datetime.datetime.now()
    filename='model_flex_'+\
    str(args.task)+'_'+\
    args.covariate+'_'+args.modelname+'_'+hier+'_'+str(now.year)+\
    '_'+str(now.month)+'_'+str(now.day)+'_'+str(args.steps)+'_'+str(args.seed)+str(args.subset)+'_'+mask_name+'_cdm'

    # Save empty placeholder
    with open('./model_fits/'+filename+'.pkl', "wb" ) as buff:
        pickle.dump({}, buff)


    # Fitting model
    with model:

        #MAP = pm.find_MAP() # Don't need to re-run
        MAP = {}

        step=pm.HamiltonianMC(target_accept=.95)

        print('sampling from posterior')
        trace = pm.sample(args.steps,step=step,chains=4,tune=args.steps_tune,random_seed=args.seed) # cores = 4

        ppc = pm.sample_ppc(trace,500)


    with open('./model_fits/'+filename+'.pkl', "wb" ) as buff:
        pickle.dump({'model': model,'trace':trace,'ppc':ppc,'MAP':MAP}, buff)

if __name__=='__main__':
    main()
