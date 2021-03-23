import sys
import pymc3 as pm
import theano
import theano.tensor as T

def calc_choice_val_sep(estimate_r_t,
                        mag_1_t,
                        mag_0_t,
                        Bm,
                        Bp):
    '''Using separate Betas for mag and prob'''
    mag_diff = (mag_1_t-mag_0_t)
    prob_diff =(estimate_r_t-(1.0-estimate_r_t))

    return(Bm*mag_diff + Bp*prob_diff)


def calc_choice_val_sep_w_baseline(estimate_r_t,
                        mag_1_t,
                        mag_0_t,
                        Bbase,
                        Bpoverm):
    '''Using separate Betas for mag and prob'''
    mag_diff = (mag_1_t-mag_0_t)
    prob_diff =(estimate_r_t-(1.0-estimate_r_t))

    return(Bbase*mag_diff + (Bbase+Bpoverm)*prob_diff)

def calc_choice_val_w_mix(estimate_r_t,
                        mag_1_t,
                        mag_0_t,
                        Binv,
                        Amix):
    '''Using separate Betas for mag and prob'''
    mag_diff = (mag_1_t-mag_0_t)
    prob_diff =(estimate_r_t-(1.0-estimate_r_t))
    return(Binv*((1-Amix)*mag_diff + (Amix)*prob_diff))

def calc_choice_val_Q_w_mix(estimate_r_t_A,
                      estimate_r_t_B,
                        mag_1_t,
                        mag_0_t,
                        Binv,
                        Amix):
    '''Using separately estimated probs'''
    mag_diff = (mag_1_t-mag_0_t)
    prob_diff =(estimate_r_t_A-estimate_r_t_B)*2
    # these are about 1/2 the size of the ones in prob diff-(1-probdiff)

    return(Binv*((1-Amix)*mag_diff + (Amix)*prob_diff))

def calc_choice_val_ev(estimate_r_t,
                        mag_1_t,
                        mag_0_t,
                        inv_tmp_t):
    '''Using inverse temperature and softmax'''
    ev_diff = (estimate_r_t*mag_1_t - (1.0-estimate_r_t)*mag_0_t)

    return(inv_tmp_t*ev_diff)

def add_pers_to_choice_value(choice_value,Bc,prev_choices_t):
    choice_value+=Bc*prev_choices_t
    return(choice_value)

def calc_p_choice(choice_value):
    return(1.0/(1.0+T.exp(-1.0*choice_value)))

def add_eps_to_choice(p_choice,eps):
    return(eps*0.5+(1.0-eps)*p_choice)
