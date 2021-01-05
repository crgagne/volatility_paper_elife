import sys
import sys
import os

import pymc3 as pm
import theano
import theano.tensor as T

import numpy as np
import pickle
import imp
import sys
import pandas as pd
import datetime
import copy

def invlogit(p):
    return 1 / (1 + np.exp(-p))

# my imports
sys.path.append('../')
sys.path.append('../model_code/')
sys.path.append('../data_processing_code/')

import get_data
from get_data import get_data

import model_base
from model_base import *

import argparse

def main():
    '''Example:
    python fit_model_to_generated_dataset.py --modelname "11" --exp 1 --steps 1000 --steps_tune 100 --seed 3

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed','-se', type=int, default=3)
    parser.add_argument('--modelname', '-m', type=str, default=None)
    parser.add_argument('--steps', '-st', type=int, default=1000)
    parser.add_argument('--steps_tune', '-stt', type=int, default=100)
    parser.add_argument('--task','-tt',type=str,default='both')
    parser.add_argument('--exp', '-e', type=int, default=1)

    args = parser.parse_args()
    print(args.steps)
    print(args.steps_tune)
    print(args.seed)
    print(type(args.seed))
    print(args.exp)

    # load behavioral data
    if args.exp==1:
        dftmp = pd.read_csv('../data/participant_table_exp1.csv')
        data = get_data(dftmp)
    else:
        dftmp = pd.read_csv('../data/participant_table_exp2.csv')
        data = get_data_online(dftmp)

    # set up data for model fitting (extract relevant behavioral data)
    X = {}
    Y = {}
    C = {}
    subj_indices = slice(0,157)
    subj_indices_86 = slice(0,86)
    X['NN']=data['outcomes_c_flipped'].shape[1]
    X['Nboth']=data['Nboth']
    X['Nrewonly']=data['Nrewonly']
    X['Npainonly']=data['Npainonly']
    subj_indices_both=slice(0,X['Nboth'])
    subj_indices_rew_only=slice(0,X['Nrewonly'])
    subj_indices_pain_only=slice(0,X['Npainonly'])

    Y['participants_choice'] = data['participants_choice'][:,subj_indices]
    for var in ['outcomes_c_flipped','mag_1_c','mag_0_c','stabvol','rewpain']:
        X[var]=data[var][:,subj_indices]

    C['Bi1item_w_j_scaled_both']=data['Bi1item_w_j_scaled_both'][subj_indices_both]
    C['Bi2item_w_j_scaled_both']=data['Bi2item_w_j_scaled_both'][subj_indices_both]
    C['Bi3item_w_j_scaled_both']=data['Bi3item_w_j_scaled_both'][subj_indices_both]
    C['Bi1item_w_j_scaled_rew_only']=data['Bi1item_w_j_scaled_rew_only'][subj_indices_rew_only]
    C['Bi2item_w_j_scaled_rew_only']=data['Bi2item_w_j_scaled_rew_only'][subj_indices_rew_only]
    C['Bi3item_w_j_scaled_rew_only']=data['Bi3item_w_j_scaled_rew_only'][subj_indices_rew_only]
    C['Bi1item_w_j_scaled_pain_only']=data['Bi1item_w_j_scaled_pain_only'][subj_indices_pain_only]
    C['Bi2item_w_j_scaled_pain_only']=data['Bi2item_w_j_scaled_pain_only'][subj_indices_pain_only]
    C['Bi3item_w_j_scaled_pain_only']=data['Bi3item_w_j_scaled_pain_only'][subj_indices_pain_only]


    # load estimated parameters from actual dataset
    if args.modelname=='11':

        # some specifications for fitting
        covariate = 'Bi3itemCDM'
        hierarchical=True
        B_max = 10
        import models_2thr9_11 as model_specific

        # load previous fit / parameters
        # this file path might need to be changed, depending on how the main model was run.
        model_name ='model=11_covariate=Bi3itemCDM_date=2021_1_5_samples=1000_seed=3_exp=1.pkl'
        with open('../fitting_behavioral_model/model_fits/'+model_name, "rb" ) as buff:
            model_output = pickle.load(buff)
        trace=model_output['trace'];
        ppc=model_output['ppc']
        model=model_output['model']
        params = model.params

        # extract previous parameters, these are the ground truth parameters that we want to recover
        Theta_est = trace['Theta'].mean(axis=0)

        #subset participants (not implemented right now)
        Theta_est = Theta_est[subj_indices,:]

    if args.modelname=='11trip':

        covariate = 'Bi3itemCDM'
        hierarchical=True
        B_max = 10
        import models_2thr9_11 as model_specific

        # load previous fit / parameters
        model_name ='model=11trip_covariate=Bi3itemCDM_date=2021_1_5_samples=1000_seed=3_exp=1.pkl'
        with open('../fitting_behavioral_model/model_fits/'+model_name, "rb" ) as buff:
            model_output = pickle.load(buff)
        trace=model_output['trace'];
        ppc=model_output['ppc']
        model=model_output['model']
        params = model.params

        # extract previous parameters, these are the ground truth parameters that we want to recover
        Theta_est = Theta_est[subj_indices,:]

    # specify generative model
    f = model_specific.create_gen_choice_model(X,Y,param_names=params,B_max=B_max,seed=int(args.seed))

    # generate new data using ground truth parameters
    gen_choice,gen_outcome_valence,*_=f(Theta_est)

    # replace participants choices with generative choices
    Y_gen = {}
    Y_gen['participants_choice']=gen_choice
    X_gen = copy.deepcopy(X)
    X_gen['outcome_valence']=gen_outcome_valence # only used for visualization

    idx_first_reward_pain= np.min([pi for (pi,p) in enumerate(params) if 'rew' in p])

    # compile base model
    model = create_model_base(X_gen,Y_gen,C,
                    params=params,
                    K=len(params),
                    Konetask=idx_first_reward_pain,
                    rew_slice=slice(0,idx_first_reward_pain),
                    pain_slice=slice(0,idx_first_reward_pain),
                    split_by_reward=True,
                    includes_subjs_with_one_task=True,
                    covariate=covariate,
                    hierarchical=hierarchical,
                    covv='diag',
                    coding='deviance',
                   )

    # compile specific model
    model = model_specific.combined_prior_model_to_choice_model(X_gen,Y_gen,param_names=params,model=model,save_state_variables=False,B_max=B_max)

    # save name
    now = datetime.datetime.now()
    filename='model='+args.modelname+'_date='+str(now.year)+\
    '_'+str(now.month)+'_'+str(now.day)+'_samples='+str(args.steps)+'_seed='+str(args.seed)+'_exp='+str(args.exp)

    # save empty placeholder
    with open('./model_fits/'+filename+'.pkl', "wb" ) as buff:
        print('saving placeholder')
        pickle.dump({}, buff)

    # sample (fit)
    with model:
        print('sampling from posterior')
        MAP = {}
        step=pm.HamiltonianMC(target_accept=.95)
        trace = pm.sample(args.steps,step=step,chains=4,tune=args.steps_tune,random_seed=args.seed)
        ppc = pm.sample_ppc(trace,500)

    if hierarchical:
        hier='hier'
    else:
        hier='nonhier'

    # save completed results
    with open('./model_fits/'+filename+'.pkl', "wb" ) as buff:
        pickle.dump({'model': model,'trace':trace,'ppc':ppc,'MAP':MAP,
            'Theta_est':Theta_est,'X_gen': X_gen,'Y_gen': Y_gen,'C':C,
            'subj_indices':subj_indices,'subj_indices_both':subj_indices_both,
            'subj_indices_rew_only':subj_indices_rew_only,
            'subj_indices_pain_only':subj_indices_pain_only}, buff)


if __name__=='__main__':
    main()
