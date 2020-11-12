
import matplotlib.pyplot as plt
import seaborn as sns
from convert_params import *


name_replace_nounderscores = {
 'lr_baseline':r'baseline',
 'lr_goodbad':r'(good-bad)',
 'lr_stabvol':r'(volatile-stable)',
 'lr_goodbad_stabvol':r'(good-bad)x(volatile-stable)',
    'lr_rewpain':r'(reward-aversive)',
 'lr_rewpain_goodbad':r'(reward-aversive)x(good-bad)',
 'lr_rewpain_stabvol':r'(reward-aversive)x(volatile-stable)',
 'lr_rewpain_goodbad_stabvol':r'$(reward-aversive)x(good-bad)x(volatile-stable)$',
 'lr_c_baseline':r'$\eta_{baseline}$',
 'Amix_baseline':r'$\lambda_{baseline}$',
 'Amix_goodbad':r'$\lambda_{good-bad}$',
 'Amix_stabvol':r'$\lambda_{volatile-stable}$',
 'Amix_goodbad_stabvol':r'$\lambda_{(good-bad)x(volatile-stable)}$',
 'Binv_baseline':r'$\omega_{baseline}$',
 'Binv_goodbad':r'$\omega_{good-bad}$',
 'Binv_stabvol':r'$\omega_{volatile-stable}$',
 'Binv_goodbad_stabvol':r'$\omega_{(good-bad)x(volatile-stable)}$',
 'Bc_baseline':r'$\omega_{(k)baseline}$',
 'mag_baseline':r'$r_{baseline}$',

 'Amix_rewpain':r'$\lambda_{reward-aversive}$',
 'Amix_rewpain_goodbad':r'$\lambda_{(reward-aversive)x(good-bad)}$',
 'Amix_rewpain_stabvol':r'$\lambda_{(reward-aversive)x(volatile-stable)}$',
 'Binv_rewpain':r'$\omega_{reward-aversive}$',
 'Binv_rewpain_goodbad':r'$\omega_{(reward-aversive)x(good-bad)}$',
 'Binv_rewpain_stabvol':r'$\omega_{(reward-aversive)x(volatile-stable)}$',
 'Bc_rewpain':r'$\omega_{(k) reward-aversive}$',
 'mag_rewpain':r'$r_{reward-aversive}$',
}

name_replace_online_nounderscores = {
 'lr_baseline':r'baseline',
 'lr_goodbad':r'(good-bad)',
 'lr_stabvol':r'(volatile-stable)',
 'lr_goodbad_stabvol':r'(good-bad)x(volatile-stable)',
 'lr_rewpain':r'(gain-loss)',
 'lr_rewpain_goodbad':r'(gain-loss)x(good-bad)',
 'lr_rewpain_stabvol':r'(gain-loss)x(volatile-stable)',

}


def convert_diagnosis_to_color(diag):
    color=[]
    for d in diag:
        if d=='control_mfmri':
            color.append(sns.color_palette()[0])
        elif d=='control_cdm':
            color.append(sns.color_palette()[1])
        elif d=='MDD':
            color.append(sns.color_palette()[2])
        elif d=='GAD':
            color.append(sns.color_palette()[3])
        elif d=='amt':
            color.append(sns.color_palette()[4])
    return(color)

def swap_order(place1,place2,array):
    array_tmp = array.copy()
    array_tmp[place1]=array[place2]
    array_tmp[place2]=array[place1]
    return(array_tmp)

def plot_param_posterior_errorbars_onesubplot(
                trace=None, # data
                params=None, # model parameter names
                gp='u', # group parameter
                param = 'lr', # readable name
                online=False, # different name replacing
                scale='logit',
                flipstab=False,
                mode=False, # use posteriod mode or mean
                offset=0,
                ax=None,# plot characteristics
                color='k',
                legend=False,
                title=None,
                legendlabel='posterior mean (w/ 95% HDI)',
                ylabel=None,
                ylabelsize =10,
                xlabelsize = 10,
                titlesize = 10,
                xticklabelsize=7,
                yticklabelsize=5,
                legendsize=8,
                s_bar=5,
                elinewidth=1,
                legend_anchor=[0.45,-0.9],
                name_replace=name_replace_nounderscores):

    '''Error bar plot for parameter components for one parameter type (i.e learning rate)

       Inputs:
           ax: for a subplot of a larger figure

    '''

    # set current axis
    plt.sca(ax)

    # get the indexes for the model parameters
    pis = [pi for pi,p in enumerate(params) if param in p and param+'_c' not in p]
    piis =np.arange(len(pis))

    params_tmp = [name_replace[params[pi]]  for pi in pis]


    # change order for learning rate
    if len(pis)>4:
        params_tmp = swap_order(3,4,params_tmp)
        pis = swap_order(3,4,pis)
        params_tmp = swap_order(1,2,params_tmp)
        pis = swap_order(1,2,pis)
        params_tmp = swap_order(2,3,params_tmp)
        pis = swap_order(2,3,pis)
        params_tmp = swap_order(4,6,params_tmp)
        pis = swap_order(4,6,pis)
        params_tmp = swap_order(5,6,params_tmp)
        pis = swap_order(5,6,pis)

    for ii,(pii,pi,param) in enumerate(zip(piis,pis,params_tmp)):

        # flip stable volatile?
        if flipstab:
            if 'stab' in param:
                flip=-1
            else:
                flip=1
        else:
            flip=1

        # Use mean or mode
        if mode==False:
            mu = np.mean(trace[gp][:,pi]*flip,axis=0)
        else:
            mu = scipy.stats.mode(trace[gp][:,pi].flatten()*flip,axis=0)[0][0]

        # calculate eror bars
        std = np.std(trace[gp][:,pi],axis=0)
        interval = pm.stats.hpd(trace[gp][:,pi].flatten()*flip,alpha=0.05)
        lower2p5=interval[0]
        upper97p5=interval[1]

        if ii==(len(piis)-1):
            legendlabeltmp=legendlabel
        else:
            legendlabeltmp=None

        # error bar for group mean and HDI's # add extra dot, why?
        #plt.scatter(pii+offset,mu,color=color,s=s_bar)
        plt.errorbar(pii+offset,mu,yerr=np.array([[mu-lower2p5],
                                                  [upper97p5-mu]]),
                    color=color,label=legendlabeltmp,
                    marker='o',
                    markersize=s_bar,
                    elinewidth=elinewidth)

    # labels
    plt.xticks(piis,params_tmp,rotation=90,fontsize=xticklabelsize);
    plt.yticks(fontsize=yticklabelsize);
    plt.ylabel(ylabel,fontsize=ylabelsize)
    plt.title(title,fontsize=titlesize)

    # horizontal line
    plt.axhline(y=0,linestyle='--',color='k',linewidth=0.5);

    if legend:
        plt.legend(loc='lower center',ncol=1,bbox_to_anchor=legend_anchor,fontsize=legendsize)

    sns.despine()

def plot_param_by_cond_sep_axes_for_task(trace,
                                data,
                                model,
                                task='reward',
                                split='low',
                                dataset='clinical',
                                param = 'lr',
                                transform = 'logit',
                                pc ='u_PC1',
                                generate_codes=generate_codes_7,
                                median=False,
                                ax=None,
                                legend=True,
                                title=None,
                                mlabel=None,
                                ylabel=None,
                                scatter_offset=0,
                                ebar_offset=0,
                                violinplot=True,
                                violinside='left',
                                ylabelsize =10,
                                xlabelsize = 10,
                                titlesize = 10,
                                xticklabelsize=7,
                                yticklabelsize=5,
                                legendsize=8,
                                s_bar=5,
                                s=1,
                                elinewidth=1,
                                alpha_dist = 0.04,
                                ylims=[-0.1,1.1],
                                legend_anchor=[0.45,-0.9],
                                xbreakline=True,
                                include_triple=False,
                                include_errorbar=True,
                                participant_sel=None,
                                errorbar_as_simple_mean=False):


    plt.sca(ax)

    if participant_sel is None:
        participant_sel = np.ones(len(data['MID_all_unique'])).astype('bool')

    if ylabel is None:
        if param=='lr':
            ylabel = r'$\alpha$'
        elif param=='Amix':
            ylabel = r'$\lambda$'
        elif param=='Binv':
            ylabel = r'$\omega$'
        elif param=='B_c':
            ylable = r'$\omega_k$'

    if pc=='u_PC1':
        factor = 'general'
        if dataset=='clinical':
            factor_in_data = 'Bi1item_w_j_scaled_all_unique'
        elif dataset=='online':
            factor_in_data = 'Bi1item_w_j_scaled_all_unique'
    if pc=='u_PC2':
        factor='depression'
        factor_in_data = 'Bi2item_w_j_scaled_all_unique'
    if pc=='u_PC3':
        factor='anxiety'
        factor_in_data = 'Bi3item_w_j_scaled_all_unique'

    # get average parameter per participant
    Theta = trace['Theta'].mean(axis=0)
    if include_triple==False:
        effects =['baseline','goodbad','stabvol','goodbad_stabvol','rewpain','rewpain_goodbad','rewpain_stabvol']
    else:
        effects =['baseline','goodbad','stabvol','goodbad_stabvol','rewpain','rewpain_goodbad','rewpain_stabvol','rewpain_goodbad_stabvol']
    pis = [i for i,p in enumerate(model.params) if (param in p) and (param+'_c' not in p)]
    piis =np.arange(len(pis))
    params_tmp =[model.params[pi] for pi in pis]

    # individual subject parameters by condition
    lrs,conds = get_param_by_subj_by_cond_gbfirst(Theta,
                                                  index=pis,
                                                  effects=effects,
                                                  dataset=dataset,
                                                 transform=transform)

    # get posterior mean for +-1
    out = generate_codes(trace,
                         parammixture=False,
                         pis=pis,
                       param_trait=pc,
                       stdab=1,
                       transform=transform,
                       include_triple=include_triple)

    # do a split by factor
    if median==True:
        thresh = np.median(data[factor_in_data][participant_sel])
    else:
        thresh = 0

    high_idx = np.logical_and(data[factor_in_data]>=thresh,participant_sel)
    low_idx = np.logical_and(data[factor_in_data]<thresh,participant_sel)

    # indexes for the parameter
    if task=='reward':
        pos = np.array([0,1,2,3])
        slicee = slice(0,4)
    elif task=='aversive' or task =='loss':
        pos = np.array([4,5,6,7])
        slicee = slice(4,8)

    # task name based on dataset
    if dataset=='clinical':
        taskname=task
    else:
        if task=='reward':
            taskname='reward gain'
        else:
            taskname='reward loss'

    # some more specifications based on split
    if split=='high':
        idx=high_idx
        color = sns.color_palette()[3]
        plusminus = '+'
        eq='>'
        extra_legend = ', score='+plusminus+'1 on '+factor+' factor'
        extra_legend_scatter1 = ''
        extra_legend_scatter2 = ' (high '+factor+' factor scores)'
    elif split=='low':
        idx=low_idx
        color = sns.color_palette()[0]
        plusminus = '-'
        eq='<'
        extra_legend = ', score='+plusminus+'1 on '+factor+' factor'
        extra_legend_scatter1 = ''
        extra_legend_scatter2 = ' (low '+factor+' factor scores)'
    elif split=='mean':
        idx = np.arange(len(data[factor_in_data]))
        color='k'
        plusminus = '='
        eq='='
        extra_legend= ' for group average'
        extra_legend_scatter1 ='individual '
        extra_legend_scatter2 =''

    if violinplot:
        # distribution
        v1 = plt.violinplot(lrs[idx,slicee], points=50, positions=np.arange(len(pos)), widths=0.85,
                       showmeans=False, showextrema=False, showmedians=False)

        # making one sided
        for b in v1['bodies']:
            if violinside=='left':
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                b.set_color(color)
                b.set_alpha(alpha_dist)
            else:
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color(color)
                b.set_alpha(alpha_dist)

    # scatter individuals
    for j,i in enumerate(pos): # j is 1-4, i can be 4-8
        y= lrs[idx,i]
        x = np.ones_like(y)*j+0.1
        if i==pos[-1]:
            plt.scatter(x+scatter_offset,y,c=color,marker="x",s=s,label=extra_legend_scatter1+'participants'+extra_legend_scatter2)#+eq+'0 on '+factor+' factor')
        else:
            plt.scatter(x+scatter_offset,y,c=color,marker="x",s=s)

        if errorbar_as_simple_mean:
            if include_errorbar:
                yerr = y.std()/np.sqrt(len(y))
            else:
                yerr = 0
            plt.errorbar(j+0.1-0.1+ebar_offset,
                         y=y.mean(),
                         yerr=yerr,
                         color=color,
                         linestyle='None',marker='o',markersize=s_bar)

    if include_errorbar:
        # posterior mean estimates
        plt.errorbar(np.arange(len(pos))-0.1+ebar_offset,
                     y=out[split+'_conds'][slicee],
                     yerr=np.array(out[split+'_conds_se'][slicee]),
                     color=color,
                     label='posterior mean (w/ std)'+extra_legend,
                     linestyle='None',marker='o',markersize=s_bar)

    # X ticks
    xticks = out['conds']
    xticks =[p.replace('good','good outcome').replace('stab','stable').replace('rew','') for p in xticks]
    xticks =[p.replace('bad','bad outcome').replace('vol','volatile').replace('pain','') for p in xticks]
    if xbreakline:
        xticks =[p.replace('_','\n') for p in xticks]
    else:
        xticks =[p.replace('_',' ') for p in xticks]

    plt.xticks(np.arange(len(pos)),xticks[slicee],rotation=90,fontsize=xticklabelsize)
    sns.despine()
    plt.title(title,fontsize=titlesize)
    plt.ylabel(ylabel,fontsize=ylabelsize)
    plt.yticks(fontsize=yticklabelsize)
    plt.ylim(ylims)
    if legend:
        plt.legend(loc='lower center',ncol=1,bbox_to_anchor=legend_anchor,fontsize=legendsize)

    sns.despine()

def print_posteriors(params=None,
                     trace_dev=None,
                     which='lr',
                     group_param='u',
                     name=None,
                     roundit=3):
    if name is None:
        if group_param=='u':
            name='group'
        else:
            name='trait'
    print('posterior probability for '+name+' effects')
    print()
    for pi,param in enumerate(params):
        if which in param:
            x = trace_dev[group_param][:,pi][:].flatten()
            if 'stabvol' in param:
                x = -1*x # flipping so it's vol-stab
            P_greater = np.mean(x>0)
            interval = np.round(pm.stats.hpd(x,alpha=0.05),roundit)
            meann = np.mean(x)
            if P_greater<0.5:
                P_greater=1-P_greater
            if np.sign(interval[0])==np.sign(interval[1]):#P_greater>0.95:
                bold1='\033[1m'
                bold2='\033[0m'+'*'
            else:
                bold1=''
                bold2=''
            print(param+': \t P(theta<>0)='+bold1+str(np.round(P_greater,roundit))+bold2+
                  ' mean='+str(np.round(meann,roundit))+
                  ' \t 95%-CI ['+str(np.round(interval[0],roundit))+','+str(np.round(interval[1],roundit))+']')
