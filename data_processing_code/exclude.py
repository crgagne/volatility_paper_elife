

EXCLUDE_PAIN = ['cb1','cb5','cb14','cb65','cb66',
'cb82','cb70','cb77','cb83','c88','cb98','cb200']
EXCLUDE_REW = ['cb2','cb7','cb14','cb22',
'cb53','cb66','cb83','cb88','cb200']

def exclude_based_on_post_task(df,task='pain'):
    if task=='pain':
        exclude = EXCLUDE_PAIN
    else:
        exclude = EXCLUDE_REW

    return(df.loc[~df['MID'].isin(exclude)])
