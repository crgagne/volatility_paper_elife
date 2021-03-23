import numpy as np
import pandas as pd
import datetime
import glob
import copy
from exclude import EXCLUDE_PAIN,EXCLUDE_REW
from scipy.stats import pearsonr,spearmanr,ttest_1samp
import statsmodels.api as sm
import os

BASEDIR = '../'

##############################
### load data single subj ####
def load_data(datafile_name,exclude=None,MID=None):
    '''This is called by other functions.

    '''

    data = pd.read_csv(datafile_name)

    if exclude is not None: # remove trials completely #
        include = [trial for trial in np.arange(len(data)) if trial not in exclude]
        data = data.loc[include,:].copy()

    outcomes = data['green_outcome'].values
    mag_1 = data['green_mag'].values
    mag_0 = data['blue_mag'].values
    if np.abs(mag_1)[0]>1.0:
        mag_1 = mag_1/100.0
        mag_0 = mag_0/100.0

    if 'block' in data:
        block=data['block'].values # 1's for stable, 0 for volatile #
        if type(block[0]) is str:
            block[block=='stable']=1
            block[block=='volatile']=0

        if block[0]==1:
            order='stable_first'
        else:
            order='volatile_first'

    if 'run' in data:
        run=data['run'].values # 0,1,2 runs in some data sets, 0 in others

    # probability of outcome 1, most, if not all, models estimate this quantity
    if 'choice' in data:
        participant_choice = data['choice'].values

    outcome_del = np.zeros(len(outcomes))
    outcome_del[(outcomes==1)&(participant_choice==1)]=1
    outcome_del[(outcomes==0)&(participant_choice==0)]=1

    # try to get volatility for subject
    if 'pain' in datafile_name:
        task='pain'
    else:
        task='rew'
    if 'loss' in datafile_name:
        task='loss'

    vol = np.nan*np.ones_like(outcomes)

    return(np.vstack((outcomes.astype('float'),
                      mag_1.astype('float'),
                      mag_0.astype('float'),
                      block.astype('float'),
                      run.astype('float'),
                      participant_choice.astype('float'),
                      outcome_del.astype('float'),
                      vol.astype('float'))).T)


def get_block(array,block_idx,block=1):
    '''This is called by other functions.

    Inputs:
    --------
    array: trial x subject matrix
    block: trial x subject matrix of 0's and 1's (1=stable)

    Outputs:
    --------
    out: trial x subject matrix with fewer rows

    '''

    for c,col in enumerate(range(array.shape[1])):
        array_in_block = array[block_idx[:,col]==block,col]
        if c==0:
            out = array_in_block
        else:
            out = np.vstack((out,array_in_block))
    return(out.T)


##############################################
### load dataset from Experiment and task ####

def load_dataset(task,
    folders,
    mask=False,
    how_many='all',
    exclude=[0,1,2,3,4,5,6,7,8,9,90,91,92,93,94,95,96,97,98,99],
    MIDSin=None):
    '''
    This is called by other functions.

    '''

    outcomes = []
    mag_1 = []
    mag_0 = []
    participants_choices = []
    block = []
    outcomes_del = []
    vols = []
    MIDS = []
    dataset = []

    for folder in folders:

        files = glob.glob(folder+'*'+task+'*modelready*')
        if how_many is not 'all':
            assert type(how_many)==int
            files=files[0:how_many]
        for filee in files:
            dataset.append(folder.split('/')[-2])

            if 'mikes_beh' in filee:
                MID = filee.split('/')[-1].split('_')[-4]
            elif 'mikes_fm' in filee:
                MID = filee.split('/')[-1].split('_')[-3]
            elif '.dat' in filee:
                MID = filee.split('/')[2].split('_')[-4]
                if MID=='data':
                    MID=filee.split('/')[3].split('_')[-4]
            else:
                MID = filee.split('_')[-3]
                if ('gain' in filee) or ('loss' in filee):
                    MID = filee.split('_')[-2]

            if MIDSin is not None:
                if MID not in MIDSin:
                    skip=1
                else:
                    skip=0
            else:
                skip=0

            if skip==0:
                MIDS.append(MID)
                d = load_data(filee,exclude=exclude,MID=MID)

                # CHECK NUMBER OF NAN's (exclude people with more than .. 5?)

                outcomes.append(d[:,0])
                mag_1.append(d[:,1])
                mag_0.append(d[:,2])
                participants_choices.append(d[:,5])
                block.append(d[:,3])
                outcomes_del.append(d[:,6])
                vols.append(d[:,7])
    mag_1=np.array(mag_1).T
    mag_0=np.array(mag_0).T
    outcomes=np.array(outcomes).T
    participants_choices=np.array(participants_choices).T
    block=np.array(block).T
    outcomes_del=np.array(outcomes_del).T
    vols=np.array(vols).T
    participants_choices_missing = copy.copy(participants_choices)
    participants_choices_missing = np.isnan(participants_choices).astype('float')

    if mask:
        participants_choices[np.isnan(participants_choices)]=-999
        participants_choices= masked_values(participants_choices,-999)
    else:
        # replace nan choices with 0
        participants_choices[np.isnan(participants_choices)]=0

    participants_prev_choices = np.roll(participants_choices,1,axis=0)
    participants_prev_choices[participants_prev_choices==0]=-1

    out = {}
    out['MIDS']=MIDS
    out['dataset']=dataset
    out['mag_1']=mag_1
    out['mag_0']=mag_0
    out['outcomes']=outcomes
    out['participants_choices']=participants_choices
    out['participants_choices_missing']=participants_choices_missing
    out['participants_prev_choices']=participants_prev_choices
    out['outcomes_del']=outcomes_del
    out['vols']=vols
    out['block']=block

    # split by block
    outcomes_stable = get_block(outcomes,block,block=1)
    mag_0_stable = get_block(mag_0,block,block=1)
    mag_1_stable = get_block(mag_1,block,block=1)
    participants_choices_stable = get_block(participants_choices,block,block=1)

    outcomes_volatile = get_block(outcomes,block,block=0)
    mag_0_volatile = get_block(mag_0,block,block=0)
    mag_1_volatile = get_block(mag_1,block,block=0)
    participants_choices_volatile = get_block(participants_choices,block,block=0)

    num_no_resp_stab = np.sum(np.isnan(participants_choices_stable),axis=0)
    num_no_resp_vol = np.sum(np.isnan(participants_choices_volatile),axis=0)

    if mask:
        participants_choices_stable[np.isnan(participants_choices_stable)]=-999
        participants_choices_stable= masked_values(participants_choices_stable,-999)

        participants_choices_volatile[np.isnan(participants_choices_volatile)]=-999
        participants_choices_volatile = masked_values(participants_choices_volatile,-999)
    else:
        # replace nan choices with 0
        participants_choices_stable[np.isnan(participants_choices_stable)]=0
        participants_choices_volatile[np.isnan(participants_choices_volatile)]=0

    out['mag_1_stable']=mag_1_stable
    out['mag_1_volatile']=mag_1_volatile
    out['mag_0_stable']=mag_0_stable
    out['mag_0_volatile']=mag_0_volatile
    out['outcomes_stable']=outcomes_stable
    out['outcomes_volatile']=outcomes_volatile
    out['participants_choices_stable']=participants_choices_stable
    out['participants_choices_volatile']=participants_choices_volatile
    out['num_no_resp_vol']=num_no_resp_vol
    out['num_no_resp_stab']=num_no_resp_stab

    return(out)


##################################
### load experiment 1 dataset ####

def get_data(dftmp,gen_data_path=None):
    ''' This is called directly to load experiment 1 data.

    '''
    data ={}

    # get list of MIDS
    # exclude subjects
    MID_pain = list(dftmp.loc[dftmp.task=='pain','MID'].values)
    MID_rew = list(dftmp.loc[dftmp.task=='reward','MID'].values)

    print('pain task excluded:'+str(len(EXCLUDE_PAIN)))
    print('rew task excluded:'+str(len(EXCLUDE_REW)))

    sel_excl = np.array([0 if MID in EXCLUDE_PAIN else 1 for MID in MID_pain]).astype('bool')
    MID_pain = list(np.array(MID_pain)[sel_excl])

    sel_excl2 = np.array([0 if MID in EXCLUDE_REW else 1 for MID in MID_rew]).astype('bool')
    MID_rew = list(np.array(MID_rew)[sel_excl2])

    # Split into lists by whether they have both tasks or just 1
    MID_has_both = list(set(MID_pain).intersection(set(MID_rew)))
    print('has both: '+str(len(MID_has_both)))

    MID_pain_only = list(set(MID_pain).difference(MID_has_both))
    print('pain only:'+str(len(MID_pain_only)))

    MID_rew_only = list(set(MID_rew).difference(MID_has_both))
    print('rew only:'+str(len(MID_rew_only)))

    # sort lists of subjects (within each subset)
    MID_has_both=list(np.sort(list(MID_has_both)))
    MID_rew_only=list(np.sort(list(MID_rew_only)))
    MID_pain_only=list(np.sort(list(MID_pain_only)))

    # make a combined list
    MID_combined = MID_has_both+MID_has_both+MID_rew_only+MID_pain_only

    # MID idxs in the 157 matrix
    MID_rew_only_idx = np.array([MID_combined.index(MID) for MID in MID_rew_only])
    MID_pain_only_idx = np.array([MID_combined.index(MID) for MID in MID_pain_only])
    MID_both_idx = np.arange(0,len(MID_has_both)*2)

    data['MID_has_both']=MID_has_both
    data['MID_pain_only']=MID_pain_only
    data['MID_rew_only']=MID_rew_only
    data['MID_combined']=MID_combined
    data['MID_all_unique']=MID_has_both+MID_rew_only+MID_pain_only

    # Get data for each subset of subjects
    folders = [BASEDIR+'data/data_raw_exp1/']
    out_pain_has_both = load_dataset('pain',
        folders,
        mask=False,
        how_many='all',
        exclude=None,
        MIDSin=MID_has_both)

    folders = [BASEDIR+'data/data_raw_exp1/']
    out_pain_pain_only = load_dataset('pain',
        folders,
        mask=False,
        how_many='all',
        exclude=None,
        MIDSin=MID_pain_only)

    folders = [BASEDIR+'data/data_raw_exp1/']
    out_rew_has_both = load_dataset('rew',
        folders,
        mask=False,
        how_many='all',
        exclude=None,
        MIDSin=MID_has_both)

    folders = [BASEDIR+'data/data_raw_exp1/']
    out_rew_rew_only = load_dataset('rew',
        folders,
        mask=False,
        how_many='all',
        exclude=None,
        MIDSin=MID_rew_only)

    # Number of subjects X tasks that they have
    NN = len(MID_has_both)*2+len(MID_rew_only)+len(MID_pain_only)
    print('subj X task:'+str(NN))
    N = len(MID_has_both)+len(MID_rew_only)+len(MID_pain_only)
    print('subjs:'+str(N))

    Nboth = len(MID_has_both)
    Nrewonly = len(MID_rew_only)
    Npainonly = len(MID_pain_only)

    data['NN']=NN
    data['N']=N
    data['Nboth']=Nboth
    data['Nrewonly']=Nrewonly
    data['Npainonly']=Npainonly

    # create data matrices
    # trials X (rew_for_people_w_both,pain_for_people_w_both,rew_for_those_w_rew_only,pain_for_those_w_pain_only)
    # MIDs are sorted
    outcomes_c_flipped = np.hstack((out_rew_has_both['outcomes'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    1.0-out_pain_has_both['outcomes'][:,np.argsort(out_pain_has_both['MIDS'])],
                                    out_rew_rew_only['outcomes'][:,np.argsort(out_rew_rew_only['MIDS'])],
                                    1.0-out_pain_pain_only['outcomes'][:,np.argsort(out_pain_pain_only['MIDS'])]))

    mag_1_c = np.hstack((out_rew_has_both['mag_1'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['mag_1'][:,np.argsort(out_pain_has_both['MIDS'])],
                                    out_rew_rew_only['mag_1'][:,np.argsort(out_rew_rew_only['MIDS'])],
                                    out_pain_pain_only['mag_1'][:,np.argsort(out_pain_pain_only['MIDS'])]))

    mag_0_c = np.hstack((out_rew_has_both['mag_0'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['mag_0'][:,np.argsort(out_pain_has_both['MIDS'])],
                                    out_rew_rew_only['mag_0'][:,np.argsort(out_rew_rew_only['MIDS'])],
                                    out_pain_pain_only['mag_0'][:,np.argsort(out_pain_pain_only['MIDS'])]))

    participants_choice = np.hstack((out_rew_has_both['participants_choices'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['participants_choices'][:,np.argsort(out_pain_has_both['MIDS'])],
                                    out_rew_rew_only['participants_choices'][:,np.argsort(out_rew_rew_only['MIDS'])],
                                    out_pain_pain_only['participants_choices'][:,np.argsort(out_pain_pain_only['MIDS'])]))

    participants_choice_missing = np.hstack((out_rew_has_both['participants_choices_missing'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['participants_choices_missing'][:,np.argsort(out_pain_has_both['MIDS'])],
                                    out_rew_rew_only['participants_choices_missing'][:,np.argsort(out_rew_rew_only['MIDS'])],
                                    out_pain_pain_only['participants_choices_missing'][:,np.argsort(out_pain_pain_only['MIDS'])]))


    block = np.hstack((out_rew_has_both['block'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['block'][:,np.argsort(out_pain_has_both['MIDS'])],
                                    out_rew_rew_only['block'][:,np.argsort(out_rew_rew_only['MIDS'])],
                                    out_pain_pain_only['block'][:,np.argsort(out_pain_pain_only['MIDS'])]))

    good_outcomes_del_bin = np.hstack((out_rew_has_both['outcomes_del'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    1.0-out_pain_has_both['outcomes_del'][:,np.argsort(out_pain_has_both['MIDS'])],
                                    out_rew_rew_only['outcomes_del'][:,np.argsort(out_rew_rew_only['MIDS'])],
                                    1.0-out_pain_pain_only['outcomes_del'][:,np.argsort(out_pain_pain_only['MIDS'])]))

    good_outcomes_del_chi = good_outcomes_del_bin.copy()
    good_outcomes_del_chi[good_outcomes_del_chi==0]=-1
    data['good_outcomes_del_bin']=good_outcomes_del_bin
    data['good_outcomes_del_chi']=good_outcomes_del_chi
    data['outcomes_c_flipped']=outcomes_c_flipped
    data['mag_1_c']=mag_1_c
    data['mag_0_c']=mag_0_c
    data['participants_choice']=participants_choice
    data['participants_choice_missing']=participants_choice_missing
    data['block']=block

    # indicators for broadcasting (same shape as data )
    irew= np.hstack((np.ones((180,len(MID_has_both))),
               np.zeros((180,len(MID_has_both))),
               np.ones((180,len(MID_rew_only))),
               np.zeros((180,len(MID_pain_only)))))
    ipain = 1.0-irew
    print(irew.shape)

    istab=block.copy()
    ivol = 1.0-istab
    print(ivol.shape)
    if gen_data_path is not None: # used for parameter recovery analyses
        gen_data = pickle.load( open(gen_data_path, "rb" ) )

        data['participants_choice']=gen_data['participants_choice']
        data['good_outcome']=gen_data['good_outcome']
        data['bad_outcome']=1-gen_data['good_outcome']
        data['good_outcome_for_mag'] = np.roll(gen_data['good_outcome'],1,axis=0)
        data['bad_outcome_for_mag'] = np.roll(gen_data['bad_outcome'],1,axis=0)

        data['sampled_params_all']=gen_data['sampled_params_all']
        data['u']=gen_data['u']
        data['uPC1']=gen_data['uPC1']
        data['uPC2']=gen_data['uPC2']
        data['sigma']=gen_data['sigma']
        # adding deviance coding indicatrs (-1,1)(180x157)
        goodbad = data['good_outcome'].copy()
        goodbad[goodbad==0]=-1
        data['goodbad']=goodbad

        # adding deviance coding indicatrs (-1,1)(180x157)
        goodbad_mag = data['good_outcome_for_mag'].copy()
        goodbad_mag[goodbad_mag==0]=-1
        data['goodbad_mag']=goodbad_mag

    else:
        good_outcome = np.zeros_like(outcomes_c_flipped)
        good_outcome[((outcomes_c_flipped==1)&(participants_choice==1))]=1 # good thing happened on green and chose green
        good_outcome[((outcomes_c_flipped==0)&(participants_choice==0))]=1 # good thing happened on blue and chose blue
        bad_outcome=1.0-good_outcome
        print(good_outcome.shape)

        good_outcome_prev_trial = np.roll(good_outcome,1,axis=0)
        bad_outcome_prev_trial = np.roll(bad_outcome,1,axis=0)

        good_outcome_prev_trial2 = np.roll(good_outcome,2,axis=0)
        bad_outcome_prev_trial2 = np.roll(bad_outcome,2,axis=0)

        good_outcome_for_mag = np.roll(good_outcome,1,axis=0)
        bad_outcome_for_mag = np.roll(bad_outcome,1,axis=0)
        print(good_outcome_for_mag.shape)
        data['good_outcome']=good_outcome
        data['bad_outcome']=bad_outcome
        data['good_outcome_prev_trial']=good_outcome_prev_trial
        data['bad_outcome_prev_trial']=bad_outcome_prev_trial
        data['good_outcome_prev_trial2']=good_outcome_prev_trial2
        data['bad_outcome_prev_trial2']=bad_outcome_prev_trial2
        data['good_outcome_for_mag']=good_outcome_for_mag
        data['bad_outcome_for_mag']=bad_outcome_for_mag

        # adding deviance coding indicatrs (-1,1)(180x157)
        goodbad = data['good_outcome'].copy()
        goodbad[goodbad==0]=-1
        data['goodbad']=goodbad

        goodbad1 = data['good_outcome_prev_trial'].copy()
        goodbad1[goodbad1==0]=-1
        data['_trial']=goodbad1
        goodbad2= data['good_outcome_prev_trial2'].copy()
        goodbad2[goodbad2==0]=-1
        data['_trial2']=goodbad2

        # adding deviance coding indicatrs (-1,1)(180x157)
        goodbad_mag = data['good_outcome_for_mag'].copy()
        goodbad_mag[goodbad_mag==0]=-1
        data['goodbad_mag']=goodbad_mag

    data['irew']=irew
    data['ipain']=ipain
    data['istab']=istab
    data['ivol']=ivol

    rewpain = data['irew'].copy()
    rewpain[rewpain==0]=-1
    data['rewpain']=rewpain

    stabvol = data['istab'].copy()
    stabvol[stabvol==0]=-1
    data['stabvol']=stabvol

    # Covariates
    STAI = []
    STAIanx = []
    STAIdep = []
    MASQAS=[]
    MASQAD=[]
    MASQAA=[]
    MASQDS=[]
    PSWQ=[]
    CESD=[]
    BDI=[]
    EPQN=[]
    for MID in MID_combined:
        STAI.append(dftmp.loc[dftmp.MID==MID,'STAI_Trait'].values[0])
        STAIanx.append(dftmp.loc[dftmp.MID==MID,'STAI_Trait_anx'].values[0])
        STAIdep.append(dftmp.loc[dftmp.MID==MID,'STAI_Trait_dep'].values[0])
        MASQAS.append(dftmp.loc[dftmp.MID==MID,'MASQ.AS'].values[0])
        MASQAD.append(dftmp.loc[dftmp.MID==MID,'MASQ.AD'].values[0])
        MASQDS.append(dftmp.loc[dftmp.MID==MID,'MASQ.DS'].values[0])
        MASQAA.append(dftmp.loc[dftmp.MID==MID,'MASQ.AA'].values[0])
        PSWQ.append(dftmp.loc[dftmp.MID==MID,'PSWQ'].values[0])
        CESD.append(dftmp.loc[dftmp.MID==MID,'CESD'].values[0])
        BDI.append(dftmp.loc[dftmp.MID==MID,'BDI'].values[0])
        EPQN.append(dftmp.loc[dftmp.MID==MID,'EPQ.N'].values[0])
    STAI=np.array(STAI)
    STAIanx=np.array(STAIanx)
    STAIdep=np.array(STAIdep)
    MASQAD=np.array(MASQAD)
    MASQAS=np.array(MASQAS)
    MASQDS=np.array(MASQDS)
    MAASQAA=np.array(MASQAA)
    PSWQ=np.array(PSWQ)
    CESD=np.array(CESD)
    BDI=np.array(BDI)
    EPQN=np.array(EPQN)

    # Groups
    dftmp.loc[dftmp['group_p_c'].isnull(),'group_p_c']='nonpatient'
    group_p_c = []
    for MID in MID_combined:
        group_p_c.append(dftmp.loc[dftmp.MID==MID,'group_p_c'].values[0])
    group_p_c=np.array(group_p_c)

    group_diag = []
    for MID in MID_combined:
        diag = dftmp.loc[dftmp.MID==MID,'group_just_patients'].values[0]
        dataset = dftmp.loc[dftmp.MID==MID,'dataset'].values[0]
        if str(diag)=='nan':
            diag='control_'+dataset
        group_diag.append(diag)
    group_diag=np.array(group_diag)

    try:
        scores_df_bi3_noASI_w_janines = pd.read_csv(BASEDIR+'fitting_bifactor_model/bifactor_exp1_poly_scores_exp1.csv')

    except:
        # place holder until factor analysis is run
        print('Unable to load factor scores, please check whether they are in the right place.')
        scores_df_bi3_noASI_w_janines = pd.DataFrame(data = {'Unnamed: 0':MID_combined,
                                                        'g':np.ones(len(STAI)),
                                                        'F1.':np.ones(len(STAI)),
                                                        'F2.':np.ones(len(STAI))})

    scores_df_bi3_noASI_w_janines=scores_df_bi3_noASI_w_janines.rename(columns={'Unnamed: 0':'MID'})

    Bi1item_w_j = []
    Bi2item_w_j = []
    Bi3item_w_j = []
    Bi1item_subset = []
    Bi2item_subset = []
    Bi3item_subset = []
    Oblimin2_1 = []
    Oblimin2_2 = []
    Oblimin3_1 = []
    Oblimin3_2 = []
    Oblimin3_3 = []
    PCA_1 = []
    PCA_2 = []
    PCA_3 = []
    PCA_4 = []
    PCA_5 = []

    for MID in MID_combined:

        Bi1item_w_j.append(scores_df_bi3_noASI_w_janines.loc[scores_df_bi3_noASI_w_janines.MID==MID,'g'].values[0])
        Bi2item_w_j.append(scores_df_bi3_noASI_w_janines.loc[scores_df_bi3_noASI_w_janines.MID==MID,'F1.'].values[0])
        Bi3item_w_j.append(scores_df_bi3_noASI_w_janines.loc[scores_df_bi3_noASI_w_janines.MID==MID,'F2.'].values[0]) ##

    Bi1item_w_j = np.array(Bi1item_w_j)
    Bi2item_w_j = np.array(Bi2item_w_j)
    Bi3item_w_j = np.array(Bi3item_w_j)

    # Normalize covariates (across everyone)
    STAI_scaled = scale(STAI)
    STAIanx_scaled = scale(STAIanx)
    STAIdep_scaled = scale(STAIdep)

    PSWQ_imputed = np.copy(PSWQ)
    PSWQ_imputed[np.isnan(PSWQ_imputed)]=np.nanmean(PSWQ_imputed) # impute mean
    PSWQ_scaled = scale(PSWQ_imputed)
    BDI_imputed = np.copy(BDI)
    BDI_imputed[np.isnan(BDI_imputed)]=np.nanmean(BDI_imputed) # impute mean
    BDI_scaled = scale(BDI_imputed)
    MASQAA_imputed = np.copy(MASQAA)
    MASQAA_imputed[np.isnan(MASQAA_imputed)]=np.nanmean(MASQAA_imputed) # impute mean
    MASQAA_scaled = scale(MASQAA_imputed)
    MASQAD_imputed = np.copy(MASQAD)
    MASQAD_imputed[np.isnan(MASQAD_imputed)]=np.nanmean(MASQAD_imputed) # impute mean
    MASQAD_scaled = scale(MASQAD_imputed)

    STAI_nonscaled = np.array(STAI)
    STAIanx_nonscaled = np.array(STAIanx)
    STAIdep_nonscaled = np.array(STAIdep)
    MASQAD_nonscaled = np.array(MASQAD)#scale(MASQAD)
    MASQAA_nonscaled = np.array(MASQAA)
    MASQAS_nonscaled = np.array(MASQAS)
    MASQDS_nonscaled = np.array(MASQDS)
    PSWQ_nonscaled = np.array(PSWQ)
    BDI_nonscaled = np.array(BDI)
    CESD_nonscaled = np.array(CESD)
    EPQN_nonscaled = np.array(EPQN)

    group_p_c_indic = group_p_c.copy()
    group_p_c_indic[group_p_c_indic=='patient']=1.0
    group_p_c_indic[group_p_c_indic=='nonpatient']=0.0
    group_p_c_indic = group_p_c_indic.astype('float')

    Bi1item_w_j_scaled = scale(Bi1item_w_j)
    Bi2item_w_j_scaled = scale(Bi2item_w_j)
    Bi3item_w_j_scaled = scale(Bi3item_w_j)

    # useful splits of STAI
    STAI_scaled_both = STAI_scaled[0:len(MID_has_both)]
    STAI_scaled_rew_only = STAI_scaled[MID_rew_only_idx]
    STAI_scaled_pain_only = STAI_scaled[MID_pain_only_idx]
    STAI_scaled_all_unique = np.concatenate((STAI_scaled_both,STAI_scaled_rew_only,STAI_scaled_pain_only))

    STAIanx_scaled_both = STAIanx_scaled[0:len(MID_has_both)]
    STAIanx_scaled_rew_only = STAIanx_scaled[MID_rew_only_idx]
    STAIanx_scaled_pain_only = STAIanx_scaled[MID_pain_only_idx]
    STAIanx_scaled_all_unique = np.concatenate((STAIanx_scaled_both,STAIanx_scaled_rew_only,STAIanx_scaled_pain_only))

    STAIdep_scaled_both = STAIdep_scaled[0:len(MID_has_both)]
    STAIdep_scaled_rew_only = STAIdep_scaled[MID_rew_only_idx]
    STAIdep_scaled_pain_only = STAIdep_scaled[MID_pain_only_idx]
    STAIdep_scaled_all_unique = np.concatenate((STAIdep_scaled_both,STAIdep_scaled_rew_only,STAIdep_scaled_pain_only))

    PSWQ_scaled_both = PSWQ_scaled[0:len(MID_has_both)]
    PSWQ_scaled_rew_only = PSWQ_scaled[MID_rew_only_idx]
    PSWQ_scaled_pain_only = PSWQ_scaled[MID_pain_only_idx]
    PSWQ_scaled_all_unique = np.concatenate((PSWQ_scaled_both,PSWQ_scaled_rew_only,PSWQ_scaled_pain_only))

    BDI_scaled_both = BDI_scaled[0:len(MID_has_both)]
    BDI_scaled_rew_only = BDI_scaled[MID_rew_only_idx]
    BDI_scaled_pain_only = BDI_scaled[MID_pain_only_idx]
    BDI_scaled_all_unique = np.concatenate((BDI_scaled_both,BDI_scaled_rew_only,BDI_scaled_pain_only))

    MASQAD_scaled_both = MASQAD_scaled[0:len(MID_has_both)]
    MASQAD_scaled_rew_only = MASQAD_scaled[MID_rew_only_idx]
    MASQAD_scaled_pain_only = MASQAD_scaled[MID_pain_only_idx]
    MASQAD_scaled_all_unique = np.concatenate((MASQAD_scaled_both,MASQAD_scaled_rew_only,MASQAD_scaled_pain_only))

    MASQAA_scaled_both = MASQAA_scaled[0:len(MID_has_both)]
    MASQAA_scaled_rew_only = MASQAA_scaled[MID_rew_only_idx]
    MASQAA_scaled_pain_only = MASQAA_scaled[MID_pain_only_idx]
    MASQAA_scaled_all_unique = np.concatenate((MASQAA_scaled_both,MASQAA_scaled_rew_only,MASQAA_scaled_pain_only))

    STAI_nonscaled_both = STAI_nonscaled[0:len(MID_has_both)]
    STAI_nonscaled_rew_only = STAI_nonscaled[MID_rew_only_idx]
    STAI_nonscaled_pain_only = STAI_nonscaled[MID_pain_only_idx]
    STAI_nonscaled_all_unique = np.concatenate((STAI_nonscaled_both,STAI_nonscaled_rew_only,STAI_nonscaled_pain_only))

    STAIanx_nonscaled_both = STAIanx_nonscaled[0:len(MID_has_both)]
    STAIanx_nonscaled_rew_only = STAIanx_nonscaled[MID_rew_only_idx]
    STAIanx_nonscaled_pain_only = STAIanx_nonscaled[MID_pain_only_idx]
    STAIanx_nonscaled_all_unique = np.concatenate((STAIanx_nonscaled_both,STAIanx_nonscaled_rew_only,STAIanx_nonscaled_pain_only))

    STAIdep_nonscaled_both = STAIdep_nonscaled[0:len(MID_has_both)]
    STAIdep_nonscaled_rew_only = STAIdep_nonscaled[MID_rew_only_idx]
    STAIdep_nonscaled_pain_only = STAIdep_nonscaled[MID_pain_only_idx]
    STAIdep_nonscaled_all_unique = np.concatenate((STAIdep_nonscaled_both,STAIdep_nonscaled_rew_only,STAIdep_nonscaled_pain_only))

    MASQAD_nonscaled_both = MASQAD_nonscaled[0:len(MID_has_both)]
    MASQAD_nonscaled_rew_only = MASQAD_nonscaled[MID_rew_only_idx]
    MASQAD_nonscaled_pain_only = MASQAD_nonscaled[MID_pain_only_idx]
    MASQAD_nonscaled_all_unique = np.concatenate((MASQAD_nonscaled_both,MASQAD_nonscaled_rew_only,MASQAD_nonscaled_pain_only))

    MASQAA_nonscaled_both = MASQAA_nonscaled[0:len(MID_has_both)]
    MASQAA_nonscaled_rew_only = MASQAA_nonscaled[MID_rew_only_idx]
    MASQAA_nonscaled_pain_only = MASQAA_nonscaled[MID_pain_only_idx]
    MASQAA_nonscaled_all_unique = np.concatenate((MASQAA_nonscaled_both,MASQAA_nonscaled_rew_only,MASQAA_nonscaled_pain_only))

    MASQAS_nonscaled_both = MASQAS_nonscaled[0:len(MID_has_both)]
    MASQAS_nonscaled_rew_only = MASQAS_nonscaled[MID_rew_only_idx]
    MASQAS_nonscaled_pain_only = MASQAS_nonscaled[MID_pain_only_idx]
    MASQAS_nonscaled_all_unique = np.concatenate((MASQAS_nonscaled_both,MASQAS_nonscaled_rew_only,MASQAS_nonscaled_pain_only))

    MASQDS_nonscaled_both = MASQDS_nonscaled[0:len(MID_has_both)]
    MASQDS_nonscaled_rew_only = MASQDS_nonscaled[MID_rew_only_idx]
    MASQDS_nonscaled_pain_only = MASQDS_nonscaled[MID_pain_only_idx]
    MASQDS_nonscaled_all_unique = np.concatenate((MASQDS_nonscaled_both,MASQDS_nonscaled_rew_only,MASQDS_nonscaled_pain_only))

    CESD_nonscaled_both = CESD_nonscaled[0:len(MID_has_both)]
    CESD_nonscaled_rew_only = CESD_nonscaled[MID_rew_only_idx]
    CESD_nonscaled_pain_only = CESD_nonscaled[MID_pain_only_idx]
    CESD_nonscaled_all_unique = np.concatenate((CESD_nonscaled_both,CESD_nonscaled_rew_only,CESD_nonscaled_pain_only))

    PSWQ_nonscaled_both = PSWQ_nonscaled[0:len(MID_has_both)]
    PSWQ_nonscaled_rew_only = PSWQ_nonscaled[MID_rew_only_idx]
    PSWQ_nonscaled_pain_only = PSWQ_nonscaled[MID_pain_only_idx]
    PSWQ_nonscaled_all_unique = np.concatenate((PSWQ_nonscaled_both,PSWQ_nonscaled_rew_only,PSWQ_nonscaled_pain_only))

    BDI_nonscaled_both = BDI_nonscaled[0:len(MID_has_both)]
    BDI_nonscaled_rew_only = BDI_nonscaled[MID_rew_only_idx]
    BDI_nonscaled_pain_only = BDI_nonscaled[MID_pain_only_idx]
    BDI_nonscaled_all_unique = np.concatenate((BDI_nonscaled_both,BDI_nonscaled_rew_only,BDI_nonscaled_pain_only))

    EPQN_nonscaled_both = EPQN_nonscaled[0:len(MID_has_both)]
    EPQN_nonscaled_rew_only = EPQN_nonscaled[MID_rew_only_idx]
    EPQN_nonscaled_pain_only = EPQN_nonscaled[MID_pain_only_idx]
    EPQN_nonscaled_all_unique = np.concatenate((EPQN_nonscaled_both,EPQN_nonscaled_rew_only,EPQN_nonscaled_pain_only))

    group_p_c_indic_both = group_p_c_indic[0:len(MID_has_both)]
    group_p_c_indic_rew_only = group_p_c_indic[MID_rew_only_idx]
    group_p_c_indic_pain_only = group_p_c_indic[MID_pain_only_idx]
    group_p_c_indic_all_unique = np.concatenate((group_p_c_indic_both,group_p_c_indic_rew_only,group_p_c_indic_pain_only))

    group_diag_both = group_diag[0:len(MID_has_both)]
    group_diag_rew_only = group_diag[MID_rew_only_idx]
    group_diag_pain_only = group_diag[MID_pain_only_idx]
    group_diag_all_unique = np.concatenate((group_diag_both,group_diag_rew_only,group_diag_pain_only))

    # useful splits of other covariates
    Bi1item_w_j_scaled_both = Bi1item_w_j_scaled[0:len(MID_has_both)]
    Bi1item_w_j_scaled_rew_only = Bi1item_w_j_scaled[MID_rew_only_idx]
    Bi1item_w_j_scaled_pain_only = Bi1item_w_j_scaled[MID_pain_only_idx]
    Bi1item_w_j_scaled_all_unique = np.concatenate((Bi1item_w_j_scaled_both,
        Bi1item_w_j_scaled_rew_only,Bi1item_w_j_scaled_pain_only))

    Bi2item_w_j_scaled_both = Bi2item_w_j_scaled[0:len(MID_has_both)]
    Bi2item_w_j_scaled_rew_only = Bi2item_w_j_scaled[MID_rew_only_idx]
    Bi2item_w_j_scaled_pain_only = Bi2item_w_j_scaled[MID_pain_only_idx]
    Bi2item_w_j_scaled_all_unique = np.concatenate((Bi2item_w_j_scaled_both,
        Bi2item_w_j_scaled_rew_only,Bi2item_w_j_scaled_pain_only))

    Bi3item_w_j_scaled_both = Bi3item_w_j_scaled[0:len(MID_has_both)]
    Bi3item_w_j_scaled_rew_only = Bi3item_w_j_scaled[MID_rew_only_idx]
    Bi3item_w_j_scaled_pain_only = Bi3item_w_j_scaled[MID_pain_only_idx]
    Bi3item_w_j_scaled_all_unique = np.concatenate((Bi3item_w_j_scaled_both,
        Bi3item_w_j_scaled_rew_only,Bi3item_w_j_scaled_pain_only))

    data['STAI_scaled']=STAI_scaled
    data['STAI_scaled_both']=STAI_scaled_both
    data['STAI_scaled_pain_only']=STAI_scaled_pain_only
    data['STAI_scaled_rew_only']=STAI_scaled_rew_only
    data['STAI_scaled_all_unique']=STAI_scaled_all_unique

    data['STAIanx_scaled']=STAIanx_scaled
    data['STAIanx_scaled_both']=STAIanx_scaled_both
    data['STAIanx_scaled_pain_only']=STAIanx_scaled_pain_only
    data['STAIanx_scaled_rew_only']=STAIanx_scaled_rew_only
    data['STAIanx_scaled_all_unique']=STAIanx_scaled_all_unique

    data['STAIdep_scaled']=STAIdep_scaled
    data['STAIdep_scaled_both']=STAIdep_scaled_both
    data['STAIdep_scaled_pain_only']=STAIdep_scaled_pain_only
    data['STAIdep_scaled_rew_only']=STAIdep_scaled_rew_only
    data['STAIdep_scaled_all_unique']=STAIdep_scaled_all_unique

    data['PSWQ_scaled']=PSWQ_scaled
    data['PSWQ_scaled_both']=PSWQ_scaled_both
    data['PSWQ_scaled_pain_only']=PSWQ_scaled_pain_only
    data['PSWQ_scaled_rew_only']=PSWQ_scaled_rew_only
    data['PSWQ_scaled_all_unique']=PSWQ_scaled_all_unique

    data['MASQAD_scaled']=MASQAD_scaled
    data['MASQAD_scaled_both']=MASQAD_scaled_both
    data['MASQAD_scaled_pain_only']=MASQAD_scaled_pain_only
    data['MASQAD_scaled_rew_only']=MASQAD_scaled_rew_only
    data['MASQAD_scaled_all_unique']=MASQAD_scaled_all_unique

    data['MASQAA_scaled']=MASQAA_scaled
    data['MASQAA_scaled_both']=MASQAA_scaled_both
    data['MASQAA_scaled_pain_only']=MASQAA_scaled_pain_only
    data['MASQAA_scaled_rew_only']=MASQAA_scaled_rew_only
    data['MASQAA_scaled_all_unique']=MASQAA_scaled_all_unique

    data['BDI_scaled']=BDI_scaled
    data['BDI_scaled_both']=BDI_scaled_both
    data['BDI_scaled_pain_only']=BDI_scaled_pain_only
    data['BDI_scaled_rew_only']=BDI_scaled_rew_only
    data['BDI_scaled_all_unique']=BDI_scaled_all_unique

    data['STAI_nonscaled']=STAI_nonscaled
    data['STAI_nonscaled_all_unique']=STAI_nonscaled_all_unique
    data['STAIanx_nonscaled_all_unique']=STAIanx_nonscaled_all_unique
    data['STAIdep_nonscaled_all_unique']=STAIdep_nonscaled_all_unique
    data['MASQAA_nonscaled_all_unique']=MASQAA_nonscaled_all_unique
    data['MASQAD_nonscaled_all_unique']=MASQAD_nonscaled_all_unique
    data['MASQAS_nonscaled_all_unique']=MASQAS_nonscaled_all_unique
    data['MASQDS_nonscaled_all_unique']=MASQDS_nonscaled_all_unique
    data['PSWQ_nonscaled_all_unique']=PSWQ_nonscaled_all_unique
    data['CESD_nonscaled_all_unique']=CESD_nonscaled_all_unique
    data['BDI_nonscaled_all_unique']=BDI_nonscaled_all_unique
    data['EPQN_nonscaled_all_unique']=EPQN_nonscaled_all_unique

    data['group_diag']=group_diag
    data['group_diag_both']=group_diag_both
    data['group_diag_pain_only']=group_diag_pain_only
    data['group_diag_rew_only']=group_diag_rew_only
    data['group_diag_all_unique']=group_diag_all_unique

    data['group_p_c_indic']=group_p_c_indic
    data['group_p_c_indic_both']=group_p_c_indic_both
    data['group_p_c_indic_pain_only']=group_p_c_indic_pain_only
    data['group_p_c_indic_rew_only']=group_p_c_indic_rew_only
    data['group_p_c_indic_all_unique']=group_p_c_indic_all_unique

    group_p_c_dindic=group_p_c_indic.copy()
    group_p_c_dindic[group_p_c_dindic==0]=-1
    group_p_c_dindic_both=group_p_c_indic_both.copy()
    group_p_c_dindic_both[group_p_c_dindic_both==0]=-1
    group_p_c_dindic_pain_only=group_p_c_indic_pain_only.copy()
    group_p_c_dindic_pain_only[group_p_c_dindic_pain_only==0]=-1
    group_p_c_dindic_rew_only=group_p_c_indic_rew_only.copy()
    group_p_c_dindic_rew_only[group_p_c_dindic_rew_only==0]=-1
    group_p_c_dindic_all_unique=group_p_c_indic_all_unique.copy()
    group_p_c_dindic_all_unique[group_p_c_dindic_all_unique==0]=-1
    data['group_p_c_dindic']=group_p_c_dindic
    data['group_p_c_dindic_both']=group_p_c_dindic_both
    data['group_p_c_dindic_pain_only']=group_p_c_dindic_pain_only
    data['group_p_c_dindic_rew_only']=group_p_c_dindic_rew_only
    data['group_p_c_dindic_all_unique']=group_p_c_dindic_all_unique

    data['Bi1item_w_j_scaled']=Bi1item_w_j_scaled
    data['Bi1item_w_j_scaled_both']=Bi1item_w_j_scaled_both
    data['Bi1item_w_j_scaled_pain_only']=Bi1item_w_j_scaled_pain_only
    data['Bi1item_w_j_scaled_rew_only']=Bi1item_w_j_scaled_rew_only
    data['Bi1item_w_j_scaled_all_unique']=Bi1item_w_j_scaled_all_unique

    data['Bi2item_w_j_scaled']=Bi2item_w_j_scaled
    data['Bi2item_w_j_scaled_both']=Bi2item_w_j_scaled_both
    data['Bi2item_w_j_scaled_pain_only']=Bi2item_w_j_scaled_pain_only
    data['Bi2item_w_j_scaled_rew_only']=Bi2item_w_j_scaled_rew_only
    data['Bi2item_w_j_scaled_all_unique']=Bi2item_w_j_scaled_all_unique

    data['Bi3item_w_j_scaled']=Bi3item_w_j_scaled
    data['Bi3item_w_j_scaled_both']=Bi3item_w_j_scaled_both
    data['Bi3item_w_j_scaled_pain_only']=Bi3item_w_j_scaled_pain_only
    data['Bi3item_w_j_scaled_rew_only']=Bi3item_w_j_scaled_rew_only
    data['Bi3item_w_j_scaled_all_unique']=Bi3item_w_j_scaled_all_unique

    # Create questionnaires with g residualized out.
    scales = ['PSWQ_nonscaled_all_unique','MASQAA_nonscaled_all_unique','MASQAD_nonscaled_all_unique',
                'BDI_nonscaled_all_unique','STAIanx_nonscaled_all_unique','STAI_nonscaled_all_unique',
                'STAIdep_nonscaled_all_unique','CESD_nonscaled_all_unique','EPQN_nonscaled_all_unique']
    scales_stems = ['PSWQ','MASQAA','MASQAD','BDI','STAIanx','STAI','STAIdep','CESD','EPQN']

    # temp indicators
    tmp_rew_only_idx = np.array([np.where(np.array(data['MID_all_unique'])==MID)[0][0] for MID in data['MID_rew_only']])
    tmp_pain_only_idx = np.array([np.where(np.array(data['MID_all_unique'])==MID)[0][0] for MID in data['MID_pain_only']])

    for q,q_stem in zip(scales,scales_stems):
        y = data[q]
        y[np.isnan(y)]=np.nanmean(y) # replace 5 missing with mean.

        # residualize the general factor
        x = data['Bi1item_w_j_scaled_all_unique'] # general factor
        X = sm.add_constant(x)
        results = sm.OLS(y,X).fit()
        e = scale(results.resid) # standardize residuals
        data[q_stem+'_scaled_residG_all_unique']=e
        data[q_stem+'_scaled_residG_both']=e[0:len(MID_has_both)]
        data[q_stem+'_scaled_residG_rew_only']=e[tmp_rew_only_idx]
        data[q_stem+'_scaled_residG_pain_only']=e[tmp_pain_only_idx]

    return(data)

##################################
### load experiment 2 dataset ####

def get_data_online(dftmp):
    ''' This is called directly to load experiment 2 data.

    '''
    data = {}

    MID_has_both =list(dftmp.MID)

    MID_has_both=list(np.sort(list(MID_has_both)))

    MID_combined = MID_has_both+MID_has_both

    data ={}
    data['MID_has_both']=MID_has_both
    data['MID_combined']=MID_combined
    data['MID_all_unique']=MID_has_both # because only has both

    folders = [BASEDIR+'data/data_raw_exp2/']
    out_pain_has_both = load_dataset('loss',
        folders,
        mask=False,
        how_many='all',
        exclude=None,
        MIDSin=MID_has_both)

    folders = [BASEDIR+'data/data_raw_exp2/']
    out_rew_has_both = load_dataset('gain',
        folders,
        mask=False,
        how_many='all',
        exclude=None,
        MIDSin=MID_has_both)

    # Number of subjects X tasks that they have
    NN = len(MID_has_both)*2
    print('subj X task:'+str(NN))
    N = len(MID_has_both)
    print('subjs:'+str(N))

    Nboth = len(MID_has_both)

    data['NN']=NN
    data['N']=N
    data['Nboth']=Nboth

    # create data matrices
    # trials X (rew_for_people_w_both,pain_for_people_w_both,rew_for_those_w_rew_only,pain_for_those_w_pain_only)
    # MIDs are sorted
    outcomes_c_flipped = np.hstack((out_rew_has_both['outcomes'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    1.0-out_pain_has_both['outcomes'][:,np.argsort(out_pain_has_both['MIDS'])]))

    mag_1_c = np.hstack((out_rew_has_both['mag_1'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['mag_1'][:,np.argsort(out_pain_has_both['MIDS'])]))

    mag_0_c = np.hstack((out_rew_has_both['mag_0'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['mag_0'][:,np.argsort(out_pain_has_both['MIDS'])]))

    participants_choice = np.hstack((out_rew_has_both['participants_choices'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['participants_choices'][:,np.argsort(out_pain_has_both['MIDS'])]))

    participants_choice_missing = np.hstack((out_rew_has_both['participants_choices_missing'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['participants_choices_missing'][:,np.argsort(out_pain_has_both['MIDS'])]))


    block = np.hstack((out_rew_has_both['block'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    out_pain_has_both['block'][:,np.argsort(out_pain_has_both['MIDS'])]))

    good_outcomes_del_bin = np.hstack((out_rew_has_both['outcomes_del'][:,np.argsort(out_rew_has_both['MIDS'])],
                                    1.0-out_pain_has_both['outcomes_del'][:,np.argsort(out_pain_has_both['MIDS'])]))


    good_outcomes_del_chi = good_outcomes_del_bin.copy()
    good_outcomes_del_chi[good_outcomes_del_chi==0]=-1
    data['good_outcomes_del_bin']=good_outcomes_del_bin
    data['good_outcomes_del_chi']=good_outcomes_del_chi
    data['outcomes_c_flipped']=outcomes_c_flipped
    data['mag_1_c']=mag_1_c
    data['mag_0_c']=mag_0_c
    data['participants_choice']=participants_choice
    data['participants_choice_missing']=participants_choice_missing
    data['block']=block

    for i in range(data['mag_0_c'].shape[1]):
        if np.max(np.abs(data['mag_0_c'][:,i]))>1.0:
            data['mag_0_c'][:,i]=data['mag_0_c'][:,i]/100.0
        if np.max(np.abs(data['mag_1_c'][:,i]))>1.0:
            data['mag_1_c'][:,i]=data['mag_1_c'][:,i]/100.0

    # indicators for broadcasting (same shape as data )
    irew= np.hstack((np.ones((180,len(MID_has_both))),
               np.zeros((180,len(MID_has_both)))))

    ipain = 1.0-irew
    print(irew.shape)

    istab=block.copy()
    ivol = 1.0-istab
    print(ivol.shape)

    good_outcome = np.zeros_like(outcomes_c_flipped)
    good_outcome[((outcomes_c_flipped==1)&(participants_choice==1))]=1 # good thing happened on green and chose green
    good_outcome[((outcomes_c_flipped==0)&(participants_choice==0))]=1 # good thing happened on blue and chose blue
    bad_outcome=1.0-good_outcome
    print(good_outcome.shape)

    good_outcome_prev_trial = np.roll(good_outcome,1,axis=0)
    bad_outcome_prev_trial = np.roll(bad_outcome,1,axis=0)

    good_outcome_prev_trial2 = np.roll(good_outcome,2,axis=0)
    bad_outcome_prev_trial2 = np.roll(bad_outcome,2,axis=0)

    good_outcome_for_mag = np.roll(good_outcome,1,axis=0)
    bad_outcome_for_mag = np.roll(bad_outcome,1,axis=0)
    print(good_outcome_for_mag.shape)
    data['good_outcome']=good_outcome
    data['bad_outcome']=bad_outcome
    data['good_outcome_prev_trial']=good_outcome_prev_trial
    data['bad_outcome_prev_trial']=bad_outcome_prev_trial
    data['good_outcome_prev_trial2']=good_outcome_prev_trial2
    data['bad_outcome_prev_trial2']=bad_outcome_prev_trial2
    data['good_outcome_for_mag']=good_outcome_for_mag
    data['bad_outcome_for_mag']=bad_outcome_for_mag

    # adding deviance coding indicatrs (-1,1)(180x157)
    goodbad = data['good_outcome'].copy()
    goodbad[goodbad==0]=-1
    data['goodbad']=goodbad

    goodbad1 = data['good_outcome_prev_trial'].copy()
    goodbad1[goodbad1==0]=-1
    data['_trial']=goodbad1
    goodbad2= data['good_outcome_prev_trial2'].copy()
    goodbad2[goodbad2==0]=-1
    data['_trial2']=goodbad2

    # adding deviance coding indicatrs (-1,1)(180x157)
    goodbad_mag = data['good_outcome_for_mag'].copy()
    goodbad_mag[goodbad_mag==0]=-1
    data['goodbad_mag']=goodbad_mag

    data['irew']=irew
    data['ipain']=ipain
    data['istab']=istab
    data['ivol']=ivol

    rewpain = data['irew'].copy()
    rewpain[rewpain==0]=-1
    data['rewpain']=rewpain

    stabvol = data['istab'].copy()
    stabvol[stabvol==0]=-1
    data['stabvol']=stabvol

    # load in factor analysis stuff
    try:
        scores_df_bi3_noASI_w_janines = pd.read_csv(BASEDIR+'fitting_bifactor_model/bifactor_exp1_poly_scores_exp2.csv')
    except:
        # place holder until factor analysis is run
        print('Unable to load factor scores, please check whether they are in the right place.')
        MID_combined_temp  = ['X'+MID if len(MID)==4 else MID for MID in MID_combined]
        scores_df_bi3_noASI_w_janines = pd.DataFrame(data = {'Unnamed: 0':MID_combined_temp,
                                                        'g':np.ones(len(MID_combined)),
                                                        'F1.':np.ones(len(MID_combined)),
                                                        'F2.':np.ones(len(MID_combined))})

    scores_df_bi3_noASI_w_janines=scores_df_bi3_noASI_w_janines.rename(columns={'Unnamed: 0':'MID'})

    # Covariates
    STAI = []
    STAIanx = []
    STAIdep = []
    MASQAS=[]
    MASQAD=[]
    MASQAA=[]
    MASQDS=[]
    PSWQ=[]
    CESD=[]
    BDI=[]
    EPQN=[]

    Bi1item_w_j = []
    Bi2item_w_j = []
    Bi3item_w_j = []


    for MID in MID_combined:
        STAI.append(dftmp.loc[dftmp.MID==MID,'STAI_Trait'].values[0])
        STAIanx.append(dftmp.loc[dftmp.MID==MID,'STAI_Trait_anx'].values[0])
        STAIdep.append(dftmp.loc[dftmp.MID==MID,'STAI_Trait_dep'].values[0])
        MASQAS.append(dftmp.loc[dftmp.MID==MID,'MASQ.AS'].values[0])
        MASQAD.append(dftmp.loc[dftmp.MID==MID,'MASQ.AD'].values[0])
        MASQDS.append(dftmp.loc[dftmp.MID==MID,'MASQ.DS'].values[0])
        MASQAA.append(dftmp.loc[dftmp.MID==MID,'MASQ.AA'].values[0])
        PSWQ.append(dftmp.loc[dftmp.MID==MID,'PSWQ'].values[0])
        CESD.append(dftmp.loc[dftmp.MID==MID,'CESD'].values[0])
        BDI.append(dftmp.loc[dftmp.MID==MID,'BDI'].values[0])
        EPQN.append(dftmp.loc[dftmp.MID==MID,'EPQ.N'].values[0])

        if len(MID)==4:
            MID = 'X'+MID

        try:
            Bi1item_w_j.append(scores_df_bi3_noASI_w_janines.loc[scores_df_bi3_noASI_w_janines.MID==MID,'g'].values[0])
            Bi2item_w_j.append(scores_df_bi3_noASI_w_janines.loc[scores_df_bi3_noASI_w_janines.MID==MID,'F1.'].values[0])
            Bi3item_w_j.append(scores_df_bi3_noASI_w_janines.loc[scores_df_bi3_noASI_w_janines.MID==MID,'F2.'].values[0]) ##
        except:
            import pdb; pdb.set_trace()

    STAI=np.array(STAI)
    STAIanx=np.array(STAIanx)
    STAIdep=np.array(STAIdep)
    MASQAD=np.array(MASQAD)
    MASQAS=np.array(MASQAS)
    MASQDS=np.array(MASQDS)
    MAASQAA=np.array(MASQAA)
    PSWQ=np.array(PSWQ)
    CESD=np.array(CESD)
    BDI=np.array(BDI)
    EPQN=np.array(EPQN)

    Bi1item_w_j = np.array(Bi1item_w_j)
    Bi2item_w_j = np.array(Bi2item_w_j)
    Bi3item_w_j = np.array(Bi3item_w_j)

    # Normalize covariates (across everyone)
    STAI_scaled = scale(STAI.astype('float'))
    STAIanx_scaled = scale(STAIanx.astype('float'))
    STAIdep_scaled = scale(STAIdep.astype('float'))
    PSWQ_scaled = scale(PSWQ.astype('float'))
    PSWQ_scaled[np.isnan(PSWQ_scaled)]=np.nanmean(PSWQ_scaled) # impute mean
    BDI_scaled = scale(BDI.astype('float'))
    MASQAA_scaled = scale(np.array(MASQAA).astype('float'))
    MASQAD_scaled = scale(np.array(MASQAD).astype('float'))

    STAI_nonscaled = np.array(STAI)
    STAIanx_nonscaled = np.array(STAIanx)
    STAIdep_nonscaled = np.array(STAIdep)
    MASQAD_nonscaled = np.array(MASQAD)
    MASQAA_nonscaled = np.array(MASQAA)
    MASQAS_nonscaled = np.array(MASQAS)
    MASQDS_nonscaled = np.array(MASQDS)
    PSWQ_nonscaled = np.array(PSWQ)
    BDI_nonscaled = np.array(BDI)
    CESD_nonscaled = np.array(CESD)
    EPQN_nonscaled = np.array(EPQN)

    Bi1item_w_j_scaled = scale(Bi1item_w_j)
    Bi2item_w_j_scaled = scale(Bi2item_w_j)
    Bi3item_w_j_scaled = scale(Bi3item_w_j)

    # useful splits of STAI
    STAI_scaled_both = STAI_scaled[0:len(MID_has_both)]
    STAIanx_scaled_both = STAIanx_scaled[0:len(MID_has_both)]
    STAIdep_scaled_both = STAIdep_scaled[0:len(MID_has_both)]
    PSWQ_scaled_both = PSWQ_scaled[0:len(MID_has_both)]
    BDI_scaled_both = BDI_scaled[0:len(MID_has_both)]
    MASQAD_scaled_both = MASQAD_scaled[0:len(MID_has_both)]
    MASQAA_scaled_both = MASQAA_scaled[0:len(MID_has_both)]

    STAI_nonscaled_both = STAI_nonscaled[0:len(MID_has_both)]
    STAIanx_nonscaled_both = STAIanx_nonscaled[0:len(MID_has_both)]
    STAIdep_nonscaled_both = STAIdep_nonscaled[0:len(MID_has_both)]
    MASQAD_nonscaled_both = MASQAD_nonscaled[0:len(MID_has_both)]
    MASQAA_nonscaled_both = MASQAA_nonscaled[0:len(MID_has_both)]
    MASQAS_nonscaled_both = MASQAS_nonscaled[0:len(MID_has_both)]
    MASQDS_nonscaled_both = MASQDS_nonscaled[0:len(MID_has_both)]
    CESD_nonscaled_both = CESD_nonscaled[0:len(MID_has_both)]
    PSWQ_nonscaled_both = PSWQ_nonscaled[0:len(MID_has_both)]
    BDI_nonscaled_both = BDI_nonscaled[0:len(MID_has_both)]
    EPQN_nonscaled_both = EPQN_nonscaled[0:len(MID_has_both)]

    Bi1item_w_j_scaled_both = Bi1item_w_j_scaled[0:len(MID_has_both)]
    Bi2item_w_j_scaled_both = Bi2item_w_j_scaled[0:len(MID_has_both)]
    Bi3item_w_j_scaled_both = Bi3item_w_j_scaled[0:len(MID_has_both)]

    data['STAI']=STAI
    data['STAI_both']=STAI[0:len(MID_has_both)]
    data['STAI_scaled']=STAI_scaled
    data['STAI_scaled_both']=STAI_scaled_both
    data['STAI_nonscaled']=STAI_nonscaled
    data['STAIanx_scaled_both']=STAIanx_scaled_both
    data['STAIdep_scaled_both']=STAIdep_scaled_both
    data['STAI_nonscaled_both']=STAI_nonscaled_both
    data['STAIanx_nonscaled_both']=STAIanx_nonscaled_both
    data['STAIdep_nonscaled_both']=STAIdep_nonscaled_both
    data['MASQAD_nonscaled_both']=MASQAD_nonscaled_both
    data['MASQAD_scaled_both']=MASQAD_scaled_both
    data['MASQAS_nonscaled_both']=MASQAS_nonscaled_both
    data['MASQAA_nonscaled_both']=MASQAA_nonscaled_both
    data['MASQAA_scaled_both']=MASQAA_scaled_both
    data['MASQDS_nonscaled_both']=MASQDS_nonscaled_both
    data['PSWQ_nonscaled_both']=PSWQ_nonscaled_both
    data['PSWQ_scaled_both']=PSWQ_scaled_both
    data['CESD_nonscaled_both']=CESD_nonscaled_both
    data['BDI_nonscaled_both']=BDI_nonscaled_both
    data['BDI_scaled_both']=BDI_scaled_both
    data['EPQN_nonscaled_both']=EPQN_nonscaled_both

    data['Bi1item_w_j_scaled_both']=Bi1item_w_j_scaled_both
    data['Bi2item_w_j_scaled_both']=Bi2item_w_j_scaled_both
    data['Bi3item_w_j_scaled_both']=Bi3item_w_j_scaled_both

    data['STAI_scaled_all_unique']=STAI_scaled_both
    data['STAI_nonscaled']=STAI_nonscaled
    data['STAIanx_scaled_all_unique']=STAIanx_scaled_both
    data['STAIdep_scaled_all_unique']=STAIdep_scaled_both
    data['STAI_nonscaled_all_unique']=STAI_nonscaled_both
    data['STAIanx_nonscaled_all_unique']=STAIanx_nonscaled_both
    data['STAIdep_nonscaled_all_unique']=STAIdep_nonscaled_both
    data['MASQAD_nonscaled_all_unique']=MASQAD_nonscaled_both
    data['MASQAS_nonscaled_all_unique']=MASQAS_nonscaled_both
    data['MASQAA_nonscaled_all_unique']=MASQAA_nonscaled_both
    data['MASQDS_nonscaled_all_unique']=MASQDS_nonscaled_both
    data['PSWQ_nonscaled_all_unique']=PSWQ_nonscaled_both
    data['CESD_nonscaled_all_unique']=CESD_nonscaled_both
    data['BDI_nonscaled_all_unique']=BDI_nonscaled_both
    data['EPQN_nonscaled_all_unique']=EPQN_nonscaled_both

    data['Bi1item_w_j_scaled_all_unique']=Bi1item_w_j_scaled_both
    data['Bi2item_w_j_scaled_all_unique']=Bi2item_w_j_scaled_both
    data['Bi3item_w_j_scaled_all_unique']=Bi3item_w_j_scaled_both

    #Create questionnaires with g residualized out.
    scales = ['PSWQ_nonscaled_all_unique','MASQAA_nonscaled_all_unique','MASQAD_nonscaled_all_unique',
                'BDI_nonscaled_all_unique','STAIanx_nonscaled_all_unique','STAI_nonscaled_all_unique',
                'STAIdep_nonscaled_all_unique','CESD_nonscaled_all_unique','EPQN_nonscaled_all_unique']
    scales_stems = ['PSWQ','MASQAA','MASQAD','BDI','STAIanx','STAI','STAIdep','CESD','EPQN']
    for q,q_stem in zip(scales,scales_stems):
        y = data[q]
        y[np.isnan(y)]=np.nanmean(y) # replace 5 missing with mean.
        x = data['Bi1item_w_j_scaled_all_unique'] # general factor
        X = sm.add_constant(x)
        results = sm.OLS(y,X).fit()
        e = scale(results.resid) # standardize residuals
        data[q_stem+'_scaled_residG_all_unique']=e
        data[q_stem+'_scaled_residG_both']=e

    return data


def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    X = X.copy()
    X = np.asarray(X)
    X = X.astype('float')

    if with_mean:
        mean_ = np.nanmean(X, axis)
    if with_std:
        scale_ = np.nanstd(X, axis)

    Xr = np.rollaxis(X, axis)
    if with_mean:
        Xr -= mean_
        mean_1 = np.nanmean(Xr, axis=0)
        if not np.allclose(mean_1, 0):
            Xr -= mean_1
    if with_std:
        scale_ = scale_
        Xr /= scale_
        if with_mean:
            mean_2 = np.nanmean(Xr, axis=0)
            if not np.allclose(mean_2, 0):
                Xr -= mean_2
    return X
