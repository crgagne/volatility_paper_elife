import sys
#sys.path.append('/Users/chris/anaconda/envs/env_pymc3/lib/python3.6/')
#sys.path.append('/Users/chris/anaconda/envs/env_pymc3/lib/python3.6/site-packages/')
import imp

import os
#os.environ["THEANO_FLAGS"] = "exception_verbosity=high,optimizer=None"#"mode=FAST_RUN,device=gpu,floatX=float32"

import pymc3 as pm
imp.reload(pm)
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse
#from theano.tensor.raw_random import binomial
#srng = RandomStreams(seed=234)

trng = T.shared_randomstreams.RandomStreams(1234)


import numpy as np
import pickle
import pandas as pd

from model_base import create_sel
from choice_fun import *
from update_fun import *


#choice,outcome_valence,prob_choice,choice_val,estimate_r,
#lr,Gamma,Binv

def trial_step(info_A_tm1,info_A_t, # externally provided to function on each trial
            obs_choice_tm1,obs_choice_t,
            mag_1_t,mag_0_t,
            stabvol_t,rewpain_t,
            # outputs of this function passed back into it on next trial
            choice_tm1,# either generated or observed choice
            outcome_valence_tm1, # either generated or observed (although not used on input because immediately redefined, useful for storage)
            prob_choice_tm1, # internal state variables
            choice_val_tm1,
            estimate_tm1,
            choice_kernel_tm1,
            lr_tm1,lr_c_tm1,Gamma_tm1,Binv_tm1,Bc_tm1,mdiff_tm1,eps_tm1,
            lr_baseline,lr_goodbad,lr_stabvol,lr_rewpain, # variables accessible on all trials
            lr_goodbad_stabvol,lr_rewpain_goodbad,lr_rewpain_stabvol,
            lr_rewpain_goodbad_stabvol,
            lr_c_baseline,lr_c_goodbad,lr_c_stabvol,lr_c_rewpain, # variables accessible on all trials
            lr_c_goodbad_stabvol,lr_c_rewpain_goodbad,lr_c_rewpain_stabvol,
            lr_c_rewpain_goodbad_stabvol,
            Gamma_baseline,Gamma_goodbad,Gamma_stabvol,Gamma_rewpain,
            Gamma_goodbad_stabvol,Gamma_rewpain_goodbad,Gamma_rewpain_stabvol,
            Gamma_rewpain_goodbad_stabvol,
            Binv_baseline,Binv_goodbad,Binv_stabvol,Binv_rewpain,
            Binv_goodbad_stabvol,Binv_rewpain_goodbad,Binv_rewpain_stabvol,
            Binv_rewpain_goodbad_stabvol,
            Bc_baseline,Bc_goodbad,Bc_stabvol,Bc_rewpain,
            Bc_goodbad_stabvol,Bc_rewpain_goodbad,Bc_rewpain_stabvol,
            Bc_rewpain_goodbad_stabvol,
            mag_baseline,mag_rewpain,
            eps_baseline,eps_stabvol,eps_rewpain,eps_rewpain_stabvol,
            gen_indicator,B_max):
    '''
    Trial by Trial updates for the model

    '''

    # determine whether last trial had good outcome
    outcome_valence_tm1 = choice_tm1*info_A_tm1 +\
                 (1.0-choice_tm1)*(1.0-info_A_tm1) +\
                 (1.0-choice_tm1)*info_A_tm1*(-1.0) + \
                 (choice_tm1)*(1.0-info_A_tm1)*(-1.0)

    # determine Gamma for this trial using last good outcome
    Gamma_t = Gamma_baseline + \
        outcome_valence_tm1*Gamma_goodbad + \
        stabvol_t*Gamma_stabvol + \
        rewpain_t*Gamma_rewpain + \
        outcome_valence_tm1*stabvol_t*Gamma_goodbad_stabvol + \
        outcome_valence_tm1*rewpain_t*Gamma_rewpain_goodbad + \
        stabvol_t*rewpain_t*Gamma_rewpain_stabvol + \
        outcome_valence_tm1*stabvol_t*rewpain_t*Gamma_rewpain_goodbad_stabvol

    Gamma_t =pm.invlogit(Gamma_t)*5 # [0,5]


    # Determine Binv for this trial using last good outcome
    Binv_t = Binv_baseline + \
        outcome_valence_tm1*Binv_goodbad + \
        stabvol_t*Binv_stabvol + \
        rewpain_t*Binv_rewpain + \
        outcome_valence_tm1*stabvol_t*Binv_goodbad_stabvol + \
        outcome_valence_tm1*rewpain_t*Binv_rewpain_goodbad + \
        stabvol_t*rewpain_t*Binv_rewpain_stabvol + \
        outcome_valence_tm1*stabvol_t*rewpain_t*Binv_rewpain_goodbad_stabvol

    Binv_t = T.exp(Binv_t)
    Binv_t = T.switch(Binv_t<0.1,0.1,Binv_t )
    Binv_t = T.switch(Binv_t>B_max.value,B_max.value,Binv_t)

    # Determine Bc for this trial using last good outcome
    Bc_t = Bc_baseline + \
        outcome_valence_tm1*Bc_goodbad + \
        stabvol_t*Bc_stabvol + \
        rewpain_t*Bc_rewpain + \
        outcome_valence_tm1*stabvol_t*Bc_goodbad_stabvol + \
        outcome_valence_tm1*rewpain_t*Bc_rewpain_goodbad + \
        stabvol_t*rewpain_t*Bc_rewpain_stabvol + \
        outcome_valence_tm1*stabvol_t*rewpain_t*Bc_rewpain_goodbad_stabvol

    #Bc_t = T.exp(Bc_t)
    #Bc_t = T.switch(Bc_t<0.1,0.1,Bc_t )
    Bc_t = T.switch(Bc_t>B_max.value,B_max.value,Bc_t)
    Bc_t = T.switch(Bc_t<-1*B_max.value,-1*B_max.value,Bc_t)

    # Calculate Choice

    ev_1_t = (T.abs_(mag_1_t)**Gamma_t)*estimate_tm1
    ev_0_t = (T.abs_(mag_0_t)**Gamma_t)*(1-estimate_tm1)
    evdiff_t = (ev_1_t - ev_0_t)

    mdiff_t = (mag_1_t-mag_0_t) # not used but passed on


    cdiff_t=(choice_kernel_tm1-(1.0-choice_kernel_tm1))

    choice_val_t = Binv_t*evdiff_t + Bc_t*cdiff_t

    # before Gamma, choice value goes between -1 and 1
    prob_choice_t = 1.0/(1.0+T.exp(-1.0*choice_val_t))

    # determine eps
    eps_t = eps_baseline + \
        stabvol_t*eps_stabvol + \
        rewpain_t*eps_rewpain + \
        stabvol_t*rewpain_t*eps_rewpain_stabvol

    eps_t = pm.invlogit(eps_t)

    # add epsilon
    prob_choice_t = eps_t*0.5+(1.0-eps_t)*prob_choice_t

    # Generate choice or Copy participants choice (used for next trial as indicator)
    if gen_indicator.value==0:
        choice_t = obs_choice_t
    else:
        #import pdb; pdb.set_trace()
        #trng = T.shared_randomstreams.RandomStreams(1234)
        choice_t = trng.binomial(n=1, p=prob_choice_t,dtype='float64')
        # this works, but I don't want everyone binomial to have the same seed, so I'd want to update by 1
        #rng_val = choice_t.rng.get_value(borrow=True)   # Get the rng for rv_u
        #rng_val.seed(seed.value)                         # seeds the generator
        #choice_t.rng.set_value(rng_val, borrow=True)


    # determine whether current trial is good or bad
    outcome_valence_t = choice_t*info_A_t +\
                 (1.0-choice_t)*(1.0-info_A_t) +\
                 (1.0-choice_t)*info_A_t*(-1.0) + \
                 (choice_t)*(1.0-info_A_t)*(-1.0)

    #import pdb; pdb.set_trace()
    lr_t = lr_baseline + \
        outcome_valence_t*lr_goodbad + \
        stabvol_t*lr_stabvol + \
        rewpain_t*lr_rewpain + \
        outcome_valence_t*stabvol_t*lr_goodbad_stabvol + \
        outcome_valence_t*rewpain_t*lr_rewpain_goodbad + \
        stabvol_t*rewpain_t*lr_rewpain_stabvol + \
        outcome_valence_t*stabvol_t*rewpain_t*lr_rewpain_goodbad_stabvol

    lr_t = pm.invlogit(lr_t)

    # update probability estimate, These will be estimate after update on t
    # stored differently than before
    estimate_t = estimate_tm1 + lr_t*(info_A_t-estimate_tm1)

    # Choice kernel learning rate
    lr_c_t = lr_c_baseline + \
        outcome_valence_t*lr_c_goodbad + \
        stabvol_t*lr_c_stabvol + \
        rewpain_t*lr_c_rewpain + \
        outcome_valence_t*stabvol_t*lr_c_goodbad_stabvol + \
        outcome_valence_t*rewpain_t*lr_c_rewpain_goodbad + \
        stabvol_t*rewpain_t*lr_c_rewpain_stabvol + \
        outcome_valence_t*stabvol_t*rewpain_t*lr_c_rewpain_goodbad_stabvol

    lr_c_t = pm.invlogit(lr_c_t)

    choice_kernel_t =  choice_kernel_tm1 + lr_c_t*(choice_t - choice_kernel_tm1)

    return([choice_t,outcome_valence_t,prob_choice_t,choice_val_t,estimate_t,choice_kernel_t,lr_t,lr_c_t,Gamma_t,Binv_t,Bc_t,mdiff_t,eps_t])



def create_choice_model(X,Y,param_names,Theta,gen_indicator=0,B_max=10.0,nonlinear_indicator=0):

    '''
    Inputs:
        Theta is required to be a Tensor variable
        X is dictionary with trial data
        Y is dictionary with observed choices

    Returns (in general):
        compiled symbolic variables
            prob_choice = E(y|X)
            choice
            state variables like trial to trial learning rate
        X data is compiled with them as constant
        Theta remains symbolically associated with it.

    (if generative):
        choice has been will generatively on each trial

    (Otherwise):
        choice is copied from participants' choices

    '''

    NN = X['NN'] # number of subject_tasks

    # Generate specific parameters (verbose.. ugh)
    lr_baseline = 0; lr_goodbad = 0; lr_stabvol = 0; lr_rewpain = 0
    lr_goodbad_stabvol = 0; lr_rewpain_goodbad = 0; lr_rewpain_stabvol = 0; lr_rewpain_goodbad_stabvol = 0
    Gamma_baseline = 0; Gamma_goodbad = 0; Gamma_stabvol = 0; Gamma_rewpain = 0
    Gamma_goodbad_stabvol = 0; Gamma_rewpain_goodbad = 0; Gamma_rewpain_stabvol = 0; Gamma_rewpain_goodbad_stabvol = 0
    Binv_baseline = 0; Binv_goodbad = 0; Binv_stabvol = 0; Binv_rewpain = 0
    Binv_goodbad_stabvol = 0; Binv_rewpain_goodbad = 0; Binv_rewpain_stabvol = 0
    Binv_rewpain_goodbad_stabvol = 0
    Bc_baseline = 0; Bc_goodbad = 0; Bc_stabvol = 0; Bc_rewpain = 0
    Bc_goodbad_stabvol = 0; Bc_rewpain_goodbad = 0; Bc_rewpain_stabvol = 0
    Bc_rewpain_goodbad_stabvol = 0
    lr_c_baseline = 0; lr_c_goodbad = 0; lr_c_stabvol = 0; lr_c_rewpain = 0
    lr_c_goodbad_stabvol = 0; lr_c_rewpain_goodbad = 0; lr_c_rewpain_stabvol = 0; lr_c_rewpain_goodbad_stabvol = 0
    mag_baseline = 0;mag_rewpain = 0
    eps_baseline = -10; eps_stabvol = 0; eps_rewpain = 0
    eps_rewpain_stabvol = 0;

    for pi,param in enumerate(param_names):
        if param=='lr_baseline':
            lr_baseline = Theta[:,pi]
        if param=='lr_goodbad':
            lr_goodbad = Theta[:,pi]
        if param=='lr_stabvol':
            lr_stabvol = Theta[:,pi]
        if param=='lr_rewpain':
            lr_rewpain = Theta[:,pi]
        if param=='lr_goodbad_stabvol':
            lr_goodbad_stabvol = Theta[:,pi]
        if param=='lr_rewpain_goodbad':
            lr_rewpain_goodbad = Theta[:,pi]
        if param=='lr_rewpain_stabvol':
            lr_rewpain_stabvol = Theta[:,pi]
        if param=='lr_rewpain_goodbad_stabvol':
            lr_rewpain_goodbad_stabvol = Theta[:,pi]
        if param=='lr_c_baseline':
            lr_c_baseline = Theta[:,pi]
        if param=='lr_c_goodbad':
            lr_c_goodbad = Theta[:,pi]
        if param=='lr_c_stabvol':
            lr_c_stabvol = Theta[:,pi]
        if param=='lr_c_rewpain':
            lr_c_rewpain = Theta[:,pi]
        if param=='lr_c_goodbad_stabvol':
            lr_c_goodbad_stabvol = Theta[:,pi]
        if param=='lr_c_rewpain_goodbad':
            lr_c_rewpain_goodbad = Theta[:,pi]
        if param=='lr_c_rewpain_stabvol':
            lr_c_rewpain_stabvol = Theta[:,pi]
        if param=='lr_c_rewpain_goodbad_stabvol':
            lr_c_rewpain_goodbad_stabvol = Theta[:,pi]
        if param=='Gamma_baseline':
            Gamma_baseline = Theta[:,pi]
        if param=='Gamma_goodbad':
            Gamma_goodbad = Theta[:,pi]
        if param=='Gamma_stabvol':
            Gamma_stabvol = Theta[:,pi]
        if param=='Gamma_rewpain':
            Gamma_rewpain = Theta[:,pi]
        if param=='Gamma_goodbad_stabvol':
            Gamma_goodbad_stabvol = Theta[:,pi]
        if param=='Gamma_rewpain_goodbad':
            Gamma_rewpain_goodbad = Theta[:,pi]
        if param=='Gamma_rewpain_stabvol':
            Gamma_rewpain_stabvol = Theta[:,pi]
        if param=='Gamma_rewpain_goodbad_stabvol':
            Gamma_rewpain_goodbad_stabvol = Theta[:,pi]
        if param=='Binv_baseline':
            Binv_baseline = Theta[:,pi]
        if param=='Binv_goodbad':
            Binv_goodbad = Theta[:,pi]
        if param=='Binv_stabvol':
            Binv_stabvol = Theta[:,pi]
        if param=='Binv_rewpain':
            Binv_rewpain = Theta[:,pi]
        if param=='Binv_goodbad_stabvol':
            Binv_goodbad_stabvol = Theta[:,pi]
        if param=='Binv_rewpain_goodbad':
            Binv_rewpain_goodbad = Theta[:,pi]
        if param=='Binv_rewpain_stabvol':
            Binv_rewpain_stabvol = Theta[:,pi]
        if param=='Binv_rewpain_goodbad_stabvol':
            Binv_rewpain_goodbad_stabvol = Theta[:,pi]
        if param=='Bc_baseline':
            Bc_baseline = Theta[:,pi]
        if param=='Bc_goodbad':
            Bc_goodbad = Theta[:,pi]
        if param=='Bc_stabvol':
            Bc_stabvol = Theta[:,pi]
        if param=='Bc_rewpain':
            Bc_rewpain = Theta[:,pi]
        if param=='Bc_goodbad_stabvol':
            Bc_goodbad_stabvol = Theta[:,pi]
        if param=='Bc_rewpain_goodbad':
            Bc_rewpain_goodbad = Theta[:,pi]
        if param=='Bc_rewpain_stabvol':
            Bc_rewpain_stabvol = Theta[:,pi]
        if param=='Bc_rewpain_goodbad_stabvol':
            Bc_rewpain_goodbad_stabvol = Theta[:,pi]
        if param=='mag_baseline':
            mag_baseline = Theta[:,pi]
        if param=='mag_rewpain':
            mag_rewpain = Theta[:,pi]
        if param=='eps_baseline':
            eps_baseline = Theta[:,pi]
        if param=='eps_stabvol':
            eps_stabvol = Theta[:,pi]
        if param=='eps_rewpain':
            eps_rewpain = Theta[:,pi]
        if param=='eps_rewpain_stabvol':
            eps_rewpain_stabvol = Theta[:,pi]


    # Create starting values scan variables (what to use for first 2 iterations)
    starting_estimate_r = T.ones(NN)*0.5
    starting_choice_val = T.ones(NN)*0.0
    starting_prob_choice = T.ones(NN)*0.5
    starting_choice = T.ones(NN)#,dtype='int64')#
    starting_outcome_valence = T.ones(NN)#,dtype='int64')#
    starting_lr = T.ones(NN)*0.5
    starting_lr_c = T.ones(NN)*0.5
    starting_Gamma = T.ones(NN)*0.5
    starting_Binv = T.ones(NN)*0.5
    starting_Bc = T.ones(NN)*0.5
    starting_choice_kernel = T.ones(NN)*0.5
    starting_mdiff= T.ones(NN)*0.2
    starting_eps = T.ones(NN)*0.2
    #import pdb; pdb.set_trace()

    (choice,outcome_valence,prob_choice,choice_val,estimate_r,choice_kernel,
    lr,lr_c,Gamma,Binv,Bc,mdiff,eps), updates = theano.scan(fn=trial_step,
                                            outputs_info=[starting_choice,starting_outcome_valence, # observables
                                                starting_prob_choice,starting_choice_val,starting_estimate_r,starting_choice_kernel,
                                                starting_lr,starting_lr_c,starting_Gamma,starting_Binv,starting_Bc,starting_mdiff,starting_eps], # note that outcome c flipped are outcome A is assigned good outcome; should be called info
                                            sequences=[dict(input=T.as_tensor_variable(np.vstack((np.ones(NN),X['outcomes_c_flipped']))),taps=[-1,0]),
                                                    dict(input=T.as_tensor_variable(np.vstack((np.ones(NN),Y['participants_choice']))),taps=[-1,0]),
                                                    T.as_tensor_variable(X['mag_1_c']),
                                                    T.as_tensor_variable(X['mag_0_c']),
                                                    T.as_tensor_variable(X['stabvol']),
                                                    T.as_tensor_variable(X['rewpain'])],
                                            non_sequences=[lr_baseline,lr_goodbad,lr_stabvol,lr_rewpain,
                                                            lr_goodbad_stabvol,lr_rewpain_goodbad,lr_rewpain_stabvol,
                                                            lr_rewpain_goodbad_stabvol,
                                                            lr_c_baseline,lr_c_goodbad,lr_c_stabvol,lr_c_rewpain,
                                                            lr_c_goodbad_stabvol,lr_c_rewpain_goodbad,lr_c_rewpain_stabvol,
                                                            lr_c_rewpain_goodbad_stabvol,
                                                            Gamma_baseline,Gamma_goodbad,Gamma_stabvol,Gamma_rewpain,
                                                            Gamma_goodbad_stabvol,Gamma_rewpain_goodbad,Gamma_rewpain_stabvol,
                                                            Gamma_rewpain_goodbad_stabvol,
                                                            Binv_baseline,Binv_goodbad,Binv_stabvol,Binv_rewpain,
                                                            Binv_goodbad_stabvol,Binv_rewpain_goodbad,Binv_rewpain_stabvol,
                                                            Binv_rewpain_goodbad_stabvol,
                                                            Bc_baseline,Bc_goodbad,Bc_stabvol,Bc_rewpain,
                                                            Bc_goodbad_stabvol,Bc_rewpain_goodbad,Bc_rewpain_stabvol,
                                                            Bc_rewpain_goodbad_stabvol,
                                                            mag_baseline,mag_rewpain,
                                                            eps_baseline,eps_stabvol,eps_rewpain,
                                                            eps_rewpain_stabvol,
                                                            gen_indicator,B_max],
                                                      strict=True)


    return((choice,outcome_valence,prob_choice,choice_val,estimate_r,choice_kernel,lr,lr_c,Gamma,Binv,Bc,mdiff,eps), updates)



def combined_prior_model_to_choice_model(X,Y,param_names,model,
                save_state_variables=False,B_max=10.0,nonlinear_indicator=0):

    '''Converts base model which just has untransformed matrix of parameters, Theta,
    and creates internal state variables, like probability estimate, and attaches to observed choice
    Inputs:
        PyMC3 model
        params is list of param names
        data is my data dictionary


    Returns:
        model with specific functional form

    '''
    with model:

        # save params with it
        model.params = param_names
        model.args_specific = {'model_name':__name__,
                                'save_state_variables':save_state_variables}

        (choice,outcome_valence,
        prob_choice,choice_val,
        estimate_r,choice_kernel,lr,lr_c,Gamma,Binv,Bc,mdiff,eps), updates = create_choice_model(X,Y,param_names,model.Theta,gen_indicator=0,B_max=B_max)

        if save_state_variables:
            estimate_r = pm.Deterministic('estimate_r',estimate_r)
            choice_val = pm.Deterministic('choice_val',choice_val)
            prob_choice = pm.Deterministic('prob_choice',prob_choice)
            choice = pm.Deterministic('choice',choice)
            outcome_valence = pm.Deterministic('outcome_valence',outcome_valence)
            choice_kernel = pm.Deterministic('choice_kernel',choice_kernel)
            lr = pm.Deterministic('lr',lr)
            lr_c = pm.Deterministic('lr_c',lr_c)
            Gamma = pm.Deterministic('Gamma',Gamma)
            Binv = pm.Deterministic('Binv',Binv)
            Bc = pm.Deterministic('Bc',Bc)
            mdiff = pm.Deterministic('mdiff',mdiff)
            eps = pm.Deterministic('eps',eps)

        observed_choice = pm.Bernoulli('observed_choice',p=prob_choice,
                                                 observed=Y['participants_choice'])

    return(model)


def create_gen_choice_model(X,Y,param_names,B_max=10.0,seed=1,nonlinear_indicator=0):

    '''
    Inputs:
        Symbolic

    Returns:
        Generative model which can be called with particular Theta (as np.array)

    '''
    NN = X['NN'] # number of subject_tasks
    Theta=T.ones((NN,len(param_names)))

    (choice,outcome_valence,
    prob_choice,choice_val,
    estimate_r,choice_kernel,lr,lr_c,Gamma,Binv,Bc,mdiff,eps), updates = create_choice_model(X,Y,param_names,Theta,gen_indicator=1,B_max=B_max)

    # set random seed when compiling the function
    shared_random_stream = [u for u in updates][0] # don't know how to unpack otherwise; returns a RandomStateSharedVariable which has a random state
    rng_val = shared_random_stream.get_value(borrow=True)
    rng_val.seed(seed)
    shared_random_stream.set_value(rng_val,borrow=True)

    f = theano.function([Theta],
    [choice,outcome_valence,
    prob_choice,choice_val,
    estimate_r,choice_kernel,lr,lr_c,Gamma,Binv,Bc,mdiff,eps],updates=updates)#no_default_updates=True)

    return(f)
