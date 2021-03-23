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
from get_data import get_data, get_data_online

import model_base
from model_base import *

import argparse

def main():
    '''
    This python function is a wrapper used to fit behavioral models to data.

    Examples of how to call it:

    # Fitting Main Model to Exp 1
    python fit_model_to_dataset.py --modelname "11" --exp 1 --steps 2000 --steps_tune 100 --covariate Bi3itemCDM --seed 3

    # Fitting Main Model to Exp 2
    python fit_model_to_dataset.py --modelname "11" --exp 2 --steps 2000 --steps_tune 100 --covariate Bi3itemCDM --seed 3

    # Fitting Other Models
    python fit_model_to_dataset.py --modelname "1" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "2" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "3" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "4" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "5" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "6" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "7" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "8" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "9" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "10" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "11" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "12" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3
    python fit_model_to_dataset.py --modelname "13" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3

    # Interaction model
    python fit_model_to_dataset.py --modelname "11trip" --exp 1 --steps 1000 --steps_tune 100 --covariate Bi3itemCDM --seed 3

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--modelname', '-m', type=str, default=None)
    parser.add_argument('--steps', '-st', type=int, default=1000)
    parser.add_argument('--steps_tune', '-stt', type=int, default=100)
    parser.add_argument('--covariate', '-c', type=str, default='None')
    parser.add_argument('--exp', '-e', type=int, default=1)
    parser.add_argument('--hierarchical','-hh',type=str,default='True')
    parser.add_argument('--task','-tt',type=str,default='both')
    parser.add_argument('--subset','-sub',type=str,default='all')
    parser.add_argument('--covariatemask','-cm',type=str,default='None')

    args = parser.parse_args()
    print(args.modelname)
    print(args.steps)
    print(args.steps_tune)
    print(args.exp)
    print(args.hierarchical)
    print(args.subset)
    print(args.covariatemask)
    args.hierarchical=eval(args.hierarchical)

    if args.steps>500:
        save_state_variables=False
    else:
        save_state_variables=False

    B_max = 10
    nonlinear_indicator = 0 # mag diff scaled

    if args.task=='both':

        if args.modelname=='1':
            # Model 1 #
            import models_1_flex as model_specific
            params=['lr_baseline','lr_stabvol',
             'Gamma_baseline','Gamma_stabvol',
             'Binv_baseline','Binv_stabvol',
             'lr_rewpain','lr_rewpain_stabvol',
            'Gamma_rewpain','Gamma_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='2':
            # Model 2 #
            import models_2thr9_11 as model_specific
            params=['lr_baseline','lr_stabvol',
             'Amix_baseline','Amix_stabvol',
             'Binv_baseline','Binv_stabvol',
             'lr_rewpain','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_stabvol',
            ]
            B_max = 10


        if args.modelname=='3':
            # Model 3 #
            import models_2thr9_11 as model_specific
            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_stabvol',
             'Binv_baseline','Binv_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='4':
            # Model 4 #
            import models_2thr9_11 as model_specific
            params=['lr_baseline','lr_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'lr_rewpain','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='5':
            # Model 5 #
            import models_2thr9_11 as model_specific
            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol',
            ]
            B_max = 10

        if args.modelname=='6':
            # Model 6 #
            import models_2thr9_11 as model_specific
            params=['lr_baseline','lr_goodbad','lr_stabvol','lr_goodbad_stabvol',
             'Amix_baseline','Amix_goodbad','Amix_stabvol','Amix_goodbad_stabvol',
             'Binv_baseline','Binv_goodbad','Binv_stabvol','Binv_goodbad_stabvol',
             'lr_rewpain','lr_rewpain_goodbad','lr_rewpain_stabvol','lr_rewpain_goodbad_stabvol',
            'Amix_rewpain','Amix_rewpain_goodbad','Amix_rewpain_stabvol','Amix_rewpain_goodbad_stabvol',
            'Binv_rewpain','Binv_rewpain_goodbad','Binv_rewpain_stabvol','Binv_rewpain_goodbad_stabvol',
            ]
            B_max = 10

        if args.modelname=='7':
            # Model 7 #
            import models_2thr9_11 as model_specific
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

        if args.modelname=='8':
            # Model 8 #
            import models_2thr9_11 as model_specific
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

        if args.modelname=='9':
            # Model 9 #
            import models_9_12 as model_specific
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

        if args.modelname=='10':
            # Model 10 #
            import models_10_13 as model_specific
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

        if args.modelname=='11':
            # Model 11 MAIN MODEL #
            import models_2thr9_11 as model_specific
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

        if args.modelname=='12':
            # Model 12 #
            import models_9_12 as model_specific
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

        if args.modelname=='13':
            # Model 13 #
            import models_10_13 as model_specific
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

        if args.modelname=='11trip':
            # Model 11 with triple interaction
            import models_2thr9_11 as model_specific
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


    # load data
    if args.exp==1:
        dftmp = pd.read_csv('../data/participant_table_exp1.csv')
        data = get_data(dftmp)
    else:
        dftmp = pd.read_csv('../data/participant_table_exp2.csv')
        data = get_data_online(dftmp)

    u_covariate_mask = None
    mask_name=''

    if args.task=='both':

        if args.subset=='all':


            if args.exp==1:
                includes_subjs_with_one_task = True

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
                             'PSWQ_scaled_residG','MASQAA_scaled_residG','MASQAD_scaled_residG',
                             'BDI_scaled_residG','STAIanx_scaled_residG','STAI_scaled_residG',
                             'PSWQ_scaled','MASQAA_scaled','MASQAD_scaled',
                             'BDI_scaled','STAIanx_scaled','STAI_scaled']:
                    for tail in ['both','pain_only','rew_only']:
                        C[stem+'_'+tail]=data[stem+'_'+tail]
            elif args.exp==2:
                includes_subjs_with_one_task = False
                
                # prepare for model code
                subj_indices = slice(0,data['participants_choice'].shape[1]) #list(np.where(np.array(data['MID_combined'])=='cb100')[0])
                Nboth = data['Nboth']

                Y = {}
                Y['participants_choice'] = data['participants_choice'][:,subj_indices]

                X = {}
                for var in ['outcomes_c_flipped','mag_1_c','mag_0_c','stabvol','rewpain']:
                    X[var]=data[var][:,subj_indices]
                X['NN']=X[var].shape[1]
                X['Nboth']=data['Nboth']


                C = {}
                C['STAI_scaled_both']=data['STAI_scaled_both']
                for trait in ['Bi1item_w_j_scaled','Bi2item_w_j_scaled','Bi3item_w_j_scaled',
                                'PSWQ_scaled_residG','MASQAA_scaled_residG','MASQAD_scaled_residG',
                                'BDI_scaled_residG','STAIanx_scaled_residG','STAI_scaled_residG',
                                'PSWQ_scaled','MASQAA_scaled','MASQAD_scaled',
                                'BDI_scaled','STAIanx_scaled','STAI_scaled']:
                    C[trait+'_both']=np.array(list(data[trait+'_both']))


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
                        includes_subjs_with_one_task=includes_subjs_with_one_task,
                        covariate=args.covariate,
                        hierarchical=args.hierarchical,
                        covv='diag',
                        coding='deviance',
                        u_covariate_mask=u_covariate_mask)

    # Create likelihood model
    print('compiling specific model')
    model = model_specific.combined_prior_model_to_choice_model(X,Y,
                                                                    param_names=params,
                                                                    model=model,
                                                                    save_state_variables=save_state_variables,
                                                                    B_max=B_max,nonlinear_indicator=nonlinear_indicator)

    # Save name
    print('saving')
    now = datetime.datetime.now()
    filename='model='+args.modelname+'_covariate='+args.covariate+'_date='+str(now.year)+\
    '_'+str(now.month)+'_'+str(now.day)+'_samples='+str(args.steps)+'_seed='+str(args.seed)+'_exp='+str(args.exp)

    # Save empty placeholder
    with open('./model_fits/'+filename+'.pkl', "wb" ) as buff:
        pickle.dump({}, buff)

    # Fitting model
    with model:

        MAP = {}

        step=pm.HamiltonianMC(target_accept=.95)

        print('sampling ...')
        trace = pm.sample(args.steps,step=step,chains=4,tune=args.steps_tune,random_seed=args.seed) # cores = 4

        ppc = pm.sample_ppc(trace,500)

    with open('./model_fits/'+filename+'.pkl', "wb" ) as buff:
        pickle.dump({'model': model,'trace':trace,'ppc':ppc,'MAP':MAP}, buff)

if __name__=='__main__':
    main()
