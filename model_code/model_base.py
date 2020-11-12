import imp
import pymc3 as pm
imp.reload(pm)

import theano
import theano.tensor as T
import numpy as np
import pickle
import pandas as pd


def create_sel(param,data,coding='deviance'):

    # create broadcast selector for Bk to approate trials in trials X NN
    if coding=='deviance':
        sel=T.ones_like(T.as_tensor_variable(data['irew']))
        if 'rewpain' in param:
            sel*=data['rewpain']
        if 'stabvol' in param:
            sel*=data['stabvol']
        if 'goodbad' in param:
            if ('Bp' in param) or ('Bm' in param) or ('Binv' in param) or ('Rp' in param) or ('Bbase' in param) or ('Bpoverm' in param) or ('Amix' in param):
                sel*=data['goodbad_mag']
            elif 'lr' in param:
                sel*=data['goodbad']
    return(sel)


def create_hyper_prior_group(Nboth,Nrew,Npain,K,rew_slice,pain_slice,mean=0,std=10):

    # hyper priors for group means
    u = pm.Normal('u',mean,std,shape=(K,1))

    # broadcast by person
    u_both_broad = T.transpose(u*T.ones(Nboth)) # Nboth x K
    u_rew_only_broad = T.transpose(u[rew_slice]*T.ones(Nrew)) # Nrew x K
    u_pain_only_broad = T.transpose(u[pain_slice]*T.ones(Npain)) # Npain x K

    # make combined mean (so we can add covariates)
    u_both_total = u_both_broad
    u_rew_only_total = u_rew_only_broad
    u_pain_only_total = u_pain_only_broad
    return(u_both_total,u_rew_only_total,u_pain_only_total)

def add_covariate_to_hyper_prior(u_both_total,
                            u_rew_only_total,
                            u_pain_only_total,rew_slice,pain_slice,K,
                            covariate,C,mean=0,std=1,
                            u_covariate_mask=None,
                            includes_subjs_with_one_task=True):

    ## SINGLE COVARIATE MODELS ##
    stems_single = {"Bi1itemCDM":'Bi1item_w_j_scaled',
             'PSWQresidG': 'PSWQ_scaled_residG',
             'PSWQ': 'PSWQ_scaled',
             'MASQADresidG': 'MASQAD_scaled_residG',
             'MASQAD': 'MASQAD_scaled',
             'MASQAAresidG': 'MASQAA_scaled_residG',
             'MASQAA': 'MASQAA_scaled',
             'STAIanxresidG': 'STAIanx_scaled_residG',
             'STAIanx': 'STAIanx_scaled',
             'STAIdepresidG': 'STAIdep_scaled_residG',
             'STAIdep': 'STAIdep_scaled',
             'STAIresidG': 'STAI_scaled_residG',
             'STAI': 'STAI_scaled',
             }

    if covariate in stems_single.keys():
        stem = stems_single[covariate]

        u_PC1 = pm.Normal('u_PC1',mean,std,shape=(K,1))
        u_PC1 = u_PC1*u_covariate_mask

        # broadcast by person and
        u_both_PC1_broad = T.transpose(u_PC1*T.as_tensor_variable(C[stem+'_both']))
        if includes_subjs_with_one_task:
            u_rew_only_PC1_broad = T.transpose(u_PC1[rew_slice]*T.as_tensor_variable(C[stem+'_rew_only']))
            u_pain_only_PC1_broad = T.transpose(u_PC1[pain_slice]*T.as_tensor_variable(C[stem+'_pain_only']))

        # add to total prior mean
        u_both_total+=u_both_PC1_broad
        if includes_subjs_with_one_task:
            u_rew_only_total+=u_rew_only_PC1_broad
            u_pain_only_total+=u_pain_only_PC1_broad

    ## DOUBLE COVARIATE MODELS ##
    stems_double = {"Bi2itemCDM": ['Bi1item_w_j_scaled','Bi2item_w_j_scaled'],
             }

    if covariate in stems_double.keys():
        stem1 = stems_double[covariate][0]
        stem2 = stems_double[covariate][1]

        u_PC1 = pm.Normal('u_PC1',mean,std,shape=(K,1)) # general
        u_PC2 = pm.Normal('u_PC2',mean,std,shape=(K,1)) # anhedonia

        u_PC1 = u_PC1*u_covariate_mask
        u_PC2 = u_PC2*u_covariate_mask

        # broadcast by person and
        u_both_PC1_broad = T.transpose(u_PC1*T.as_tensor_variable(C[stem1+'_both']))
        if includes_subjs_with_one_task:
            u_rew_only_PC1_broad = T.transpose(u_PC1[rew_slice]*T.as_tensor_variable(C[stem1+'_rew_only']))
            u_pain_only_PC1_broad = T.transpose(u_PC1[pain_slice]*T.as_tensor_variable(C[stem1+'_pain_only']))

        u_both_PC2_broad = T.transpose(u_PC2*T.as_tensor_variable(C[stem2+'_both']))
        if includes_subjs_with_one_task:
            u_rew_only_PC2_broad = T.transpose(u_PC2[rew_slice]*T.as_tensor_variable(C[stem2+'_rew_only']))
            u_pain_only_PC2_broad = T.transpose(u_PC2[pain_slice]*T.as_tensor_variable(C[stem2+'_pain_only']))

        # add to total prior mean
        u_both_total+=u_both_PC1_broad
        if includes_subjs_with_one_task:
            u_rew_only_total+=u_rew_only_PC1_broad
            u_pain_only_total+=u_pain_only_PC1_broad

        u_both_total+=u_both_PC2_broad
        if includes_subjs_with_one_task:
            u_rew_only_total+=u_rew_only_PC2_broad
            u_pain_only_total+=u_pain_only_PC2_broad

    ## TRIPlE COVARIATE MODELS ##
    stems_triple = {"Bi3itemCDM": ['Bi1item_w_j_scaled','Bi2item_w_j_scaled','Bi3item_w_j_scaled'],
                    "Bi3itemCDMsubset": ['Bi1item_subset_scaled','Bi2item_subset_scaled','Bi3item_subset_scaled'],
                    "G_ADrG_PSWQrG": ['Bi1item_w_j_scaled','MASQAD_scaled_residG','PSWQ_scaled_residG'],
                    "G_ADrG_AArG": ['Bi1item_w_j_scaled','MASQAD_scaled_residG','MASQAA_scaled_residG'],
             }
    if covariate in stems_triple.keys():
        stem1 = stems_triple[covariate][0]
        stem2 = stems_triple[covariate][1]
        stem3 = stems_triple[covariate][2]

        # hyper priors for covariates
        u_PC1 = pm.Normal('u_PC1',mean,std,shape=(K,1))
        u_PC2 = pm.Normal('u_PC2',mean,std,shape=(K,1))
        u_PC3 = pm.Normal('u_PC3',mean,std,shape=(K,1))

        u_PC1 = u_PC1*u_covariate_mask
        u_PC2 = u_PC2*u_covariate_mask
        u_PC3 = u_PC3*u_covariate_mask

        # broadcast by person and
        u_both_PC1_broad = T.transpose(u_PC1*T.as_tensor_variable(C[stem1+'_both']))
        if includes_subjs_with_one_task:
            u_rew_only_PC1_broad = T.transpose(u_PC1[rew_slice]*T.as_tensor_variable(C[stem1+'_rew_only']))
            u_pain_only_PC1_broad = T.transpose(u_PC1[pain_slice]*T.as_tensor_variable(C[stem1+'_pain_only']))

        u_both_PC2_broad = T.transpose(u_PC2*T.as_tensor_variable(C[stem2+'_both']))
        if includes_subjs_with_one_task:
            u_rew_only_PC2_broad = T.transpose(u_PC2[rew_slice]*T.as_tensor_variable(C[stem2+'_rew_only']))
            u_pain_only_PC2_broad = T.transpose(u_PC2[pain_slice]*T.as_tensor_variable(C[stem2+'_pain_only']))

        u_both_PC3_broad = T.transpose(u_PC3*T.as_tensor_variable(C[stem3+'_both']))
        if includes_subjs_with_one_task:
            u_rew_only_PC3_broad = T.transpose(u_PC3[rew_slice]*T.as_tensor_variable(C[stem3+'_rew_only']))
            u_pain_only_PC3_broad = T.transpose(u_PC3[pain_slice]*T.as_tensor_variable(C[stem3+'_pain_only']))

        # add to total prior mean
        u_both_total+=u_both_PC1_broad
        if includes_subjs_with_one_task:
            u_rew_only_total+=u_rew_only_PC1_broad
            u_pain_only_total+=u_pain_only_PC1_broad

        u_both_total+=u_both_PC2_broad
        if includes_subjs_with_one_task:
            u_rew_only_total+=u_rew_only_PC2_broad
            u_pain_only_total+=u_pain_only_PC2_broad

        u_both_total+=u_both_PC3_broad
        if includes_subjs_with_one_task:
            u_rew_only_total+=u_rew_only_PC3_broad
            u_pain_only_total+=u_pain_only_PC3_broad




    return(u_both_total,u_rew_only_total,u_pain_only_total)



def create_model_base(X, # observed stuff
                Y, # choices
                C, # covariates 157x1
                K, # number of params
                Konetask, # last index of parameters that don't depend on reward pain
                rew_slice,
                pain_slice,
                params=None,
                split_by_reward=True,
                includes_subjs_with_one_task=True,
                covariate='no_covariates',
                hierarchical=True,
                covv='diag',
                coding='deviance',
                group_mean_hyper_prior_mean=0, # prior belief over group mean, can specify as T.as_tensor_variable(np.array([[0,0,0,1]]).T)
                group_mean_hyper_prior_std=10, # can specify as T.as_tensor_variable(np.array([[1,1,1,2]]).T)
                group_covariate_hyper_prior_mean=0,
                group_covariate_hyper_prior_std=1,
                theta_var_hyper_prior_std=2.5,
                theta_var_hyper_prior_dist='HalfCauchy',
                cov_ind_priors=None, # can specify for independent priors (non hierarchical) T.eye(K)*10
                u_both_total=None,  # same need pm.theanof.set_theano_conf({'compute_test_value': 'raise'})
                one_task_only=False,
                u_covariate_mask=None, # Kx1
               ):
    '''Creates a PyMC model with parameter vector Theta with
    parameters that are hierarcically distributed

    '''

    model = pm.Model()

    with model:

        model.args = {'K':K,
                    'Konetask':Konetask, # last index of parameters that don't depend on reward pain
                    'rew_slice':rew_slice, # parameter slice for rew
                    'pain_slice':pain_slice, # subject slice for pain
                    'split_by_reward':split_by_reward,
                    'covariate':covariate,
                    'hierarchical':hierarchical,
                    'covv':covv,
                    'coding':coding,
                    'params':params,
                    'group_mean_hyper_prior_mean':group_mean_hyper_prior_mean, # prior belief over group mean # be able to pass in a vector
                    'group_mean_hyper_prior_std':group_mean_hyper_prior_std,
                    'group_covariate_hyper_prior_mean':group_covariate_hyper_prior_mean,
                    'group_covariate_hyper_prior_std':group_covariate_hyper_prior_std,
                    'theta_var_hyper_prior_std':theta_var_hyper_prior_std,
                    'cov_ind_priors':cov_ind_priors,
                    'u_both_total':u_both_total,
                    'theta_var_hyper_prior_dist':theta_var_hyper_prior_dist
                    }

        NN = X['NN'] # should equal Nboth*2 + Nrewonly + Npainonly
        Nboth = X['Nboth']
        model.NN=NN
        model.Nboth=Nboth
        if includes_subjs_with_one_task:
            Nrewonly = X['Nrewonly']
            Npainonly = X['Npainonly']
            model.Nrewonly =Nrewonly
            model.Npainonly=Npainonly
        else:
            Nrewonly=0
            Npainonly=0

        if u_covariate_mask is None:
            u_covariate_mask=np.ones((len(params),1))

        if hierarchical:

            if covariate not in ['group4']:

                # Hyper Prior Means (w/ Prior distributions-belief about prior means)
                (u_both_total,
                    u_rew_only_total,
                    u_pain_only_total)=create_hyper_prior_group(Nboth,Nrewonly,Npainonly,
                        K,rew_slice,pain_slice,
                        mean=group_mean_hyper_prior_mean,
                        std=group_mean_hyper_prior_std)

                (u_both_total,u_rew_only_total,u_pain_only_total)=add_covariate_to_hyper_prior(u_both_total,
                                            u_rew_only_total,
                                            u_pain_only_total,rew_slice,pain_slice,K,
                                            covariate,C,
                                            mean=group_covariate_hyper_prior_mean,
                                            std=group_covariate_hyper_prior_std,
                                            u_covariate_mask=u_covariate_mask,
                                            includes_subjs_with_one_task=includes_subjs_with_one_task)
            else:
                (u_both_total,
                    u_rew_only_total,
                    u_pain_only_total)=create_hyper_prior_group4(C,Nboth,Nrewonly,Npainonly,
                        K,rew_slice,pain_slice,
                        mean=group_mean_hyper_prior_mean,
                        std=group_mean_hyper_prior_std)


            if covv=='diag':

                # Hyper Prior Variance
                if theta_var_hyper_prior_dist=='HalfCauchy':
                    sigma = pm.HalfCauchy('sigma',theta_var_hyper_prior_std,shape=(K,1))
                elif theta_var_hyper_prior_dist=='HalfNormal':
                    sigma = pm.HalfNormal('sigma',theta_var_hyper_prior_std,shape=(K,1))
                elif theta_var_hyper_prior_dist=='Fixed':
                    # for generating data from fixed number
                    sigma = pm.Deterministic('sigma',theta_var_hyper_prior_std)

                Sigma = T.eye(K)*(sigma)
                Sigma = pm.Deterministic('Sigma', Sigma)

            # Subject Parameters

            # Shape = N subjects with both X K parameters (usually 71 x K)
            Theta_both = pm.MvNormal('Theta_both',mu=u_both_total,shape=(Nboth,K),cov=Sigma)

            if includes_subjs_with_one_task:
                # Shape = N subjects with just reward K parameters (usually 9 x K)
                Theta_rew_only = pm.MvNormal('Theta_rew_only',mu=u_rew_only_total,shape=(Nrewonly,Konetask),cov=Sigma[rew_slice,rew_slice])

                # Shape = N subjects with just pain K parameters (usually 7 x K)
                Theta_pain_only = pm.MvNormal('Theta_pain_only',mu=u_pain_only_total,shape=(Npainonly,Konetask),cov=Sigma[pain_slice,pain_slice])


        else:
            if cov_ind_priors is None:
                cov_ind_priors = T.eye(K)*10 # Covariance Matrix Identity;
            #make the variance for parameters for non-hierarchical really large

            if u_both_total is None:
                u_both_total = T.zeros((Nboth,K)) # mean 0; shouldn't it be size K?

            Theta_both = pm.MvNormal('Theta_both',mu=u_both_total,shape=(Nboth,K),cov=cov_ind_priors)
            if includes_subjs_with_one_task:
                u_rew_only_total = T.zeros((Nrewonly,Konetask)) # recently changed this to Kone task from 1
                u_pain_only_total = T.zeros((Npainonly,Konetask))
                Theta_rew_only = pm.MvNormal('Theta_rew_only',mu=u_rew_only_total,shape=(Nrewonly,Konetask),cov=cov_ind_priors[rew_slice,rew_slice])
                Theta_pain_only = pm.MvNormal('Theta_pain_only',mu=u_pain_only_total,shape=(Npainonly,Konetask),cov=cov_ind_priors[pain_slice,pain_slice])

        if split_by_reward and includes_subjs_with_one_task:
            # add zeros for parameters for tasks that these subjects don't have
            # if not splitting the single parameter goes to which ever task the subject has
            if coding=='block':
                print('this may not work any more')
                Theta_rew_only = T.concatenate((Theta_rew_only,T.zeros_like(Theta_rew_only)),axis=1)
                Theta_pain_only = T.concatenate((T.zeros_like(Theta_pain_only),Theta_pain_only),axis=1)
            elif coding=='deviance':
                # the relevant regressors are up front for both pain task and reward task
                Theta_rew_only = T.concatenate((Theta_rew_only,T.zeros_like(Theta_rew_only)[:,0:K-Konetask]),axis=1)
                Theta_pain_only = T.concatenate((Theta_pain_only,T.zeros_like(Theta_pain_only)[:,0:K-Konetask]),axis=1)

        # Stack all together and broadcast to Subject_task instead of subjects
        # Subjects with both tasks have two copies of Theta
        # N subject_task (157) x K
        if one_task_only:
            Theta = Theta_both
        else:
            if includes_subjs_with_one_task:
                Theta = T.concatenate((Theta_both,Theta_both,Theta_rew_only,Theta_pain_only),axis=0)
            else:
                # for rew then pain
                Theta = T.concatenate((Theta_both,Theta_both),axis=0)

        Theta = pm.Deterministic('Theta',Theta)

    return(model)
