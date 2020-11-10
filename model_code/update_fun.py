
import sys
import pymc3 as pm
import theano
import theano.tensor as T

def update_estimate(next_outcome, lr,estimate_so_far):
    return (lr*(next_outcome-estimate_so_far)+estimate_so_far)

def calc_pe(outcome,estimate):
    '''useful for debugging, may not need'''
    return (outcome-estimate_so_far)

def update_Q_estimate(outcome,choice,lr,decay,estimate):
    '''
    Inputs:
        outcome is outcome on current trial
        =1 if shape rewarded or not shocked, =0 if not rewarded or shocked
        Make sure to flip outcome based on shape passed in

        lr is learning rate on current trial

        choice is the participant's choice on current trial
        =1 if shape is chosen, =0 if other shape chosen
        Make sure flip choice based on shape passed in

        estimate is the previous Q-value.

        all inputs are of size 157x0 (1 per subject in a task)

    '''

    # calculate pe
    pe = (outcome-estimate)
    regress = (0.5-estimate)

        
    # perform update or decay
    # if participant choice =1, they chose shape, so update with pe
    # if particiant choice =0, they chose other shape, so update with decay
        # otherwise return the previous estimate
    next_estimate=estimate + choice*lr*pe + (1-choice)*decay*regress

    return next_estimate

def adjust_estimate(estimate_r_t,risk_pref_t):
    # preserves TxN shape of estimates by broadcasting 1xN parameter array across the trial dimension
    estimate_r_t_adj = (estimate_r_t-0.5)*risk_pref_t+0.5

    # creates TxNx1
    estimate_r_t_adj3 = T.reshape(estimate_r_t_adj,
                                  newshape=[estimate_r_t_adj.shape[0],
                                                             estimate_r_t_adj.shape[1],1])

    # creates 0's that are TxNx1
    zeros_like_estimate_r_t_adj3 = T.reshape(T.zeros_like(estimate_r_t_adj),
                                             newshape=[estimate_r_t_adj.shape[0],
                                                             estimate_r_t_adj.shape[1],1])

    # create TxNx2 and then take the max over returning TxNx1
    estimate_r_t_adj_max = T.max(T.stack([estimate_r_t_adj3,
                                          zeros_like_estimate_r_t_adj3],
                                         axis=2),
                                 axis=2,)

    # create TxNx2 and then take the max over returning TxN
    estimate_r_t_adj_max_min=T.squeeze(T.min(T.stack([estimate_r_t_adj_max,
                                               T.ones_like(estimate_r_t_adj_max)],
                                               axis=2),
                                       axis=2))
    return(estimate_r_t_adj_max_min)
