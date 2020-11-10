
import sys
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T

import numpy as np
import pickle
import imp
import statsmodels.api as sm
import sys
import pandas as pd
import datetime


def invlogit(p):
    return 1 / (1 + np.exp(-p))

def logit(p):
    return np.log(p/(1-p))

def basecoding(gb,sv,rp):
    basecode=[0,0,0]

    if gb=='good':
        basecode[0]=1
    else:
        basecode[0]=-1

    if sv=='stab':
        basecode[1]=1
    else:
        basecode[1]=-1

    if rp=='rew':
        basecode[2]=1
    else:
        basecode[2]=-1
    return(basecode)


def get_param_by_subj_by_cond(Theta,
                              index,
                              transform='invlogit',
                              effects = [],
                              dataset='clinical'):
    '''
    Converts params from sampling space to conditions.

    Inputs:
        Theta to be point estimate so 157xK
        index of parameters in Theta like [0,1,2,4] for learning rate

    '''

    if dataset=='clinical':
        sel=np.concatenate((np.arange(0,71),np.arange(142,157))) # everyone
        param = np.zeros((86,8))
        n_subs = 86
    elif dataset=='online':
        sel=np.arange(147)
        param = np.zeros((147,8))
        n_subs = 147

    B_trace = Theta[sel,:][:,index]

    for subj in range(n_subs):
        conds = []
        ci=0
        for rp in ['rew','pain']:
            for sv in ['stab','vol']:
                for gb in ['good','bad']:
                    block = gb+'_'+sv+'_'+rp
                    basecode=basecoding(gb,sv,rp)

                    code = [] # needs to be size of the number of effects
                    for effect in effects:
                        if effect=='baseline':
                            code.append(1)
                        elif effect=='goodbad':
                            code.append(basecode[0])
                        elif effect=='stabvol':
                            code.append(basecode[1])
                        elif effect=='goodbad_stabvol':
                            code.append(basecode[0]*basecode[1])
                        elif effect=='rewpain':
                            code.append(basecode[2])
                        elif effect=='rewpain_goodbad':
                            code.append(basecode[2]*basecode[0])
                        elif effect=='rewpain_stabvol':
                            code.append(basecode[1]*basecode[2])


                    if transform=='invlogit':
                        try:
                            param[subj,ci] = (invlogit(np.sum(np.array(code)*B_trace[subj,:])))
                        except:
                            import pdb; pdb.set_trace()
                    elif transform=='exp':
                        #import pdb; pdb.set_trace()
                        param[subj,ci] = (np.exp(np.sum(np.array(code)*B_trace[subj,:])))


                    elif transform=='None':
                        param[subj,ci] = ((np.sum(np.array(code)*B_trace[subj,:])))
                    elif transform=='invlogit5':
                        try:
                            param[subj,ci] = (5*invlogit(np.sum(np.array(code)*B_trace[subj,:])))
                        except:
                            import pdb; pdb.set_trace()

                    ci+=1
                    conds.append(block)
    return(param,conds)


def get_param_by_subj_by_cond_gbfirst(Theta,
                              index,
                              transform='invlogit',
                              effects = [],
                              dataset='clinical'):
    '''
    SAME ORDER AS GENERATE CODES FUNCTION
    Converts params from sampling space to conditions.

    Inputs:
        Theta to be point estimate so 157xK
        index of parameters in Theta like [0,1,2,4] for learning rate

    '''

    if dataset=='clinical':
        sel=np.concatenate((np.arange(0,71),np.arange(142,157))) # everyone
        param = np.zeros((86,8))
        n_subs = 86
    elif dataset=='online':
        sel=np.arange(147)
        param = np.zeros((147,8))
        n_subs = 147
    elif dataset=='two_participants':
        sel = np.arange(4)
        param = np.zeros((2,8))
        n_subs = 2
    elif dataset=='two_participants_rew':
        sel = np.arange(2)
        param = np.zeros((2,8))
        n_subs = 2


    B_trace = Theta[sel,:][:,index]

    for subj in range(n_subs):
        conds = []
        ci=0
        for rp in ['rew','pain']:
            for gb in ['good','bad']:
                for sv in ['stab','vol']:
                    block = gb+'_'+sv+'_'+rp
                    basecode=basecoding(gb,sv,rp)

                    code = [] # needs to be size of the number of effects
                    for effect in effects:
                        if effect=='baseline':
                            code.append(1)
                        elif effect=='goodbad':
                            code.append(basecode[0])
                        elif effect=='stabvol':
                            code.append(basecode[1])
                        elif effect=='goodbad_stabvol':
                            code.append(basecode[0]*basecode[1])
                        elif effect=='rewpain':
                            code.append(basecode[2])
                        elif effect=='rewpain_goodbad':
                            code.append(basecode[2]*basecode[0])
                        elif effect=='rewpain_stabvol':
                            code.append(basecode[1]*basecode[2])
                        elif effect=='rewpain_goodbad_stabvol':
                            code.append(basecode[2]*basecode[0]*basecode[1])

                    #print(code)

                    if transform=='invlogit' or transform=='logit':
                        try:
                            param[subj,ci] = (invlogit(np.sum(np.array(code)*B_trace[subj,:])))
                        except:
                            import pdb; pdb.set_trace()
                    elif transform=='exp':
                        param[subj,ci] = (np.exp(np.sum(np.array(code)*B_trace[subj,:])))


                    elif transform=='None':
                        param[subj,ci] = ((np.sum(np.array(code)*B_trace[subj,:])))
                    elif transform=='invlogit5':
                        try:
                            param[subj,ci] = (5*invlogit(np.sum(np.array(code)*B_trace[subj,:])))
                        except:
                            import pdb; pdb.set_trace()

                    ci+=1
                    conds.append(block)
    return(param,conds)


def get_param_by_subj_by_cond_gbfirst_w_samples(Theta,
                              index,
                              transform='invlogit',
                              effects = [],
                              dataset='clinical'):
    '''
    UPDATE: SAME ORDER AS GENERATE CODES FUNCTION
    UPDATE: RETURNS RE-PARAMETERIZED BY SAMPLE
    Converts params from sampling space to conditions.

    Inputs:
        Theta to be point estimate so 157xK
        index of parameters in Theta like [0,1,2,4] for learning rate

    '''

    if dataset=='clinical':
        sel=np.concatenate((np.arange(0,71),np.arange(142,157))) # everyone
        param = np.zeros((Theta.shape[0],86,8))
        n_subs = 86
    elif dataset=='online':
        sel=np.arange(147)
        param = np.zeros((Theta.shape[0],147,8))
        n_subs = 147

    B_trace = Theta[:,sel][:,:,index]

    for subj in range(n_subs):
        conds = []
        ci=0
        for rp in ['rew','pain']:
            for gb in ['good','bad']:
                for sv in ['stab','vol']:
                    block = gb+'_'+sv+'_'+rp
                    basecode=basecoding(gb,sv,rp)

                    code = [] # needs to be size of the number of effects
                    for effect in effects:
                        if effect=='baseline':
                            code.append(1)
                        elif effect=='goodbad':
                            code.append(basecode[0])
                        elif effect=='stabvol':
                            code.append(basecode[1])
                        elif effect=='goodbad_stabvol':
                            code.append(basecode[0]*basecode[1])
                        elif effect=='rewpain':
                            code.append(basecode[2])
                        elif effect=='rewpain_goodbad':
                            code.append(basecode[2]*basecode[0])
                        elif effect=='rewpain_stabvol':
                            code.append(basecode[2]*basecode[1])
                        elif effect=='rewpain_goodbad_stabvol':
                            code.append(basecode[2]*basecode[0]*basecode[1])

                    for sample in range(Theta.shape[0]):
                        if transform=='invlogit' or transform=='logit':
                            param[sample,subj,ci] = (invlogit(np.sum(np.array(code)*B_trace[sample,subj,:])))
                        elif transform=='exp':
                            param[sample,subj,ci] = (np.exp(np.sum(np.array(code)*B_trace[sample,subj,:])))
                        elif transform=='None':
                            param[sample,subj,ci] = ((np.sum(np.array(code)*B_trace[sample,subj,:])))

                    ci+=1
                    conds.append(block)
    return(param,conds)


def generate_codes_7(trace,parammixture,pis=[0,1,2,3,12,13],
                   param_trait='u_bi1',
                   stdab=1,
                   transform='logit',
                   include_triple=False):


    #import pdb; pdb.set_trace()
    stai = trace[param_trait][:,pis,0]
    mu = trace['u'][:,pis,0]

    low_conds =[]
    high_conds =[]
    mean_conds = []
    low_conds_se =[]
    high_conds_se =[]
    mean_conds_se =[]
    indiv_conds=[]

    conds = []
    for rp in ['rew','pain']:
        for gb in ['good','bad']:
            for sv in ['stab','vol']:
                block = gb+'_'+sv+'_'+rp
                basecode=[0,0,0]

                if gb=='good':
                    basecode[0]=1
                else:
                    basecode[0]=-1

                if sv=='stab':
                    basecode[1]=1
                else:
                    basecode[1]=-1

                if rp=='rew':
                    basecode[2]=1
                else:
                    basecode[2]=-1

                if parammixture:
                    print('mixture parameter')
                    # mixture goes in this order
                    code = [1, # baseline
                            basecode[0], # goodbad
                            basecode[1], # stabvol
                            basecode[0]*basecode[1], # goodbad stabevol
                            basecode[2], # rewpain
                            basecode[1]*basecode[2], # stabvol rewpain
                            basecode[2]*basecode[0], # goodbad rewpain
                            ]

                else:
                    # learning rate and Binv go in this order
                    if include_triple==False:
                        code = [1, # baseline
                                basecode[0], # goodbad
                                basecode[1], # stabvol
                                basecode[0]*basecode[1], # goodbad stabevol
                                basecode[2], # rewpain
                                basecode[2]*basecode[0], # goodbad rewpain
                                basecode[1]*basecode[2]] # stabvol rewpain
                    else:
                        code = [1, # baseline
                                basecode[0], # goodbad
                                basecode[1], # stabvol
                                basecode[0]*basecode[1], # goodbad stabevol
                                basecode[2], # rewpain
                                basecode[2]*basecode[0], # goodbad rewpain
                                basecode[1]*basecode[2],
                                basecode[1]*basecode[2]*basecode[0]] # stabvol rewpain

                conds.append(block)
                if transform=='logit':
                    low_conds.append(np.mean(invlogit(np.sum(mu*code-1*stdab*stai*code,axis=1))))
                    high_conds.append(np.mean(invlogit(np.sum(mu*code+1*stdab*stai*code,axis=1))))
                    low_conds_se.append(np.std(invlogit(np.sum(mu*code-1*stdab*stai*code,axis=1))))
                    high_conds_se.append(np.std(invlogit(np.sum(mu*code+1*stdab*stai*code,axis=1))))
                    mean_conds.append(np.mean(((invlogit(np.sum(mu*code,axis=1))))))
                    mean_conds_se.append(np.std(((invlogit(np.sum(mu*code,axis=1))))))

                elif transform=='none':
                    # no logit (looks similar)
                    low_conds.append(np.mean((np.sum(mu*code-1*stdab*stai*code,axis=1))))
                    high_conds.append(np.mean((np.sum(mu*code+1*stdab*stai*code,axis=1))))
                    low_conds_se.append(np.std((np.sum(mu*code-1*stdab*stai*code,axis=1))))
                    high_conds_se.append(np.std((np.sum(mu*code+1*stdab*stai*code,axis=1))))
                    mean_conds.append(np.mean(((np.sum(mu*code,axis=1)))))
                    mean_conds_se.append(np.std(((np.sum(mu*code,axis=1)))))

                elif transform=='exp':
                    # no logit (looks similar)
                    low_conds.append(np.mean((np.exp(np.sum(mu*code-1*stdab*stai*code,axis=1)))))
                    high_conds.append(np.mean((np.exp(np.sum(mu*code+1*stdab*stai*code,axis=1)))))
                    low_conds_se.append(np.std((np.exp(np.sum(mu*code-1*stdab*stai*code,axis=1)))))
                    high_conds_se.append(np.std((np.exp(np.sum(mu*code+1*stdab*stai*code,axis=1)))))
                    mean_conds.append(np.mean(((np.exp(np.sum(mu*code,axis=1))))))
                    mean_conds_se.append(np.std(((np.exp(np.sum(mu*code,axis=1))))))
                elif transform=='logit5':
                    low_conds.append(np.mean(5*invlogit(np.sum(mu*code-1*stdab*stai*code,axis=1))))
                    high_conds.append(np.mean(5*invlogit(np.sum(mu*code+1*stdab*stai*code,axis=1))))
                    low_conds_se.append(np.std(5*invlogit(np.sum(mu*code-1*stdab*stai*code,axis=1))))
                    high_conds_se.append(np.std(5*invlogit(np.sum(mu*code+1*stdab*stai*code,axis=1))))
                    mean_conds.append(np.mean(((5*invlogit(np.sum(mu*code,axis=1))))))
                    mean_conds_se.append(np.std(((5*invlogit(np.sum(mu*code,axis=1))))))
    out={}
    out['low_conds']=low_conds
    out['high_conds']=high_conds
    out['low_conds_se']=low_conds_se
    out['high_conds_se']=high_conds_se
    out['conds']=conds
    out['mean_conds']=mean_conds
    out['mean_conds_se']=mean_conds_se

    return(out)


#### New Ways of Calculating ####


def generate_code_mat(effects):

    code_mat = []
    conds = []

    # loop through each condition
    for rp in ['rew','pain']:
        for gb in ['good','bad']:
            for sv in ['stab','vol']:
                block = gb+'_'+sv+'_'+rp
                basecode=basecoding(gb,sv,rp)

                # loop through each effect
                code = [] #
                for effect in effects:
                    if effect=='baseline':
                        code.append(1)
                    elif effect=='goodbad':
                        code.append(basecode[0])
                    elif effect=='stabvol':
                        code.append(basecode[1])
                    elif effect=='goodbad_stabvol':
                        code.append(basecode[0]*basecode[1])
                    elif effect=='rewpain':
                        code.append(basecode[2])
                    elif effect=='rewpain_goodbad':
                        code.append(basecode[2]*basecode[0])
                    elif effect=='rewpain_stabvol':
                        code.append(basecode[1]*basecode[2])
                    elif effect=='rewpain_goodbad_stabvol':
                        code.append(basecode[2]*basecode[0]*basecode[1])

                code_mat.append(code)
                conds.append(block)
    code_mat = np.array(code_mat)
    return(code_mat,conds)

def convert_params_effect_to_condition(Theta,model,effects,param = 'lr',transform='invlogit'):
    '''Theta here is a summary, e.g. posterior mean, 157xK parameters'''


    code_mat,conds = generate_code_mat(effects)

    # select the effect parameters of interest
    pis = [i for i,p in enumerate(model.params) if (param in p) and (param+'_c' not in p)]
    piis =np.arange(len(pis))
    params_tmp =[model.params[pi] for pi in pis]

    # select the data
    Theta_effect_space= Theta[:,pis]

    if transform=='invlogit':
        Theta_condition_space = invlogit(np.dot(Theta_effect_space,code_mat))
    else:
        raise NotImplemented

    return(Theta_condition_space,code_mat,conds)

def convert_params_condition_to_effect(Theta_condition_space,code_mat,transform='logit'):
    code_mat_inv = np.linalg.inv(code_mat)
    if transform=='logit':
        return(np.dot(logit(Theta_condition_space),code_mat_inv))
    else:
        raise NotImplemented
