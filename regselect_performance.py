import pickle
import os
import pprint
import time
import argparse
from functools import partial
import numpy as np
import pandas as pd
from jax import tree,vmap
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update(
    { 'font.family': 'sans-serif',
      'mathtext.default': 'regular', }
)

def color_boxplots( boxplots, color ):
    for patch in boxplots['boxes']:
        patch.set_color(color)
        patch.set_facecolor(color)
    for patch in boxplots['medians']:
        patch.set_color('k')
    for item in ['whiskers','caps']:
        # there are two whiskers and two caps.
        # we have to color the "first" and then the "second"
        # of each for each boxplot
        for patch in boxplots[item][::2]:
            patch.set_color(color) # color "first"
        for patch in boxplots[item][1::2]:
            patch.set_color(color) # color "second"
    for patch in boxplots['fliers']:
        patch.set_markeredgecolor(color)


def process_metric_trees(mnpz, model_instance, LB_factor ):
    ## how is data organized on disk?
    #
    # instance_foo_metrics.npz
    # training: tuple( cc_tree r2_tree) 
    #                  metric_tree:
    #                  recording1: metric array
    #                  recording2: metric array
    #                  ...
    #                  recordingM: metric array
    # testing  ...
    # scramble ...
    
    result = []
    for key in mnpz.files:
        for metric,metric_tree in zip( ('cc','r2'), mnpz[key] ):
            for rkey in metric_tree.keys():
                data = metric_tree[rkey]
                value = np.mean(data).item()
                result.append(
                    { 'group':key,
                      'metric':metric,
                      'model_instance':model_instance,
                      'LB_factor':LB_factor,
                      'recording':rkey,
                      'value':value}
                )
    return pd.DataFrame(result)

def process_model_instance( model_instance, LB_factor, epath ):
    rpath = os.path.join( epath, f'run{LB_factor}' )
    mpath = os.path.join(
        rpath,
        f'instance_{model_instance}_metrics.npz'
    )
    mnpz = np.load( mpath, allow_pickle=True )
    df = process_metric_trees(mnpz,model_instance,LB_factor)
    return df

def process_model_params( LB_factor, epath, num_model_instances=100 ):
    rpath = os.path.join( epath, f'run{LB_factor}' )
    print(f'processing: LB factor {LB_factor}')
    
    def load_and_cat( rpath, model_instance):
        ppath = os.path.join( rpath, f'params_{model_instance}.pkl' )
        with open( ppath, 'rb' ) as file:
            params = pickle.load( file )
        catB = np.concat( tree.leaves(params['testing']['B_tree']), axis=1 )
        return catB
    
    B_tuple = tuple( load_and_cat(rpath,model_instance) for model_instance in range( num_model_instances ) )
    return B_tuple

def run(epath,LB_factors):
    num_model_instances = 100
    df = pd.concat(
        tuple( process_model_instance( model_instance, LB_factor, epath)
               for LB_factor in LB_factors
               for model_instance in range(num_model_instances) )
    )
    return df

def regselect_boxplots(df, ax, LB_factors, metric):
    df = df[df['metric']==metric]    
    data = df.groupby(['group','LB_factor'])
    loc = np.array((-0.1,0.1))
    colors= ('r','b')
    flierprops=dict(marker='x', markersize=3)    
    groups = ('training','testing')
    for g,group in enumerate(groups):
        group_data = tuple( np.array( data.get_group((group,LB_factor)).groupby('model_instance')['value'].mean() )
                            for LB_factor in LB_factors )
        pos = loc[g]
        color=colors[g]
        l = len(group_data)
        hh=ax.boxplot( group_data,
                       patch_artist = True,
                       positions = np.arange(l)+pos,
                       showfliers=False,#flierprops=flierprops,
                       widths = 0.15)
        color_boxplots(hh,color)
    LB_factors = np.array(LB_factors)
    #labels = tuple(f'10$^{{{x}}}$' for x in -LB_factors )
    labels = tuple('' for x in -LB_factors )
    ax.set_xticks( np.arange(LB_factors.size), labels=labels )
    if metric == 'r2':
        ax.set_ylabel('Model Instance R^2')
    elif metric == 'cc':
        ax.set_ylabel('Model Instance CC')        


    LB_groups = df.groupby(['LB_factor','model_instance']).mean(numeric_only=True).groupby('LB_factor')
    # best models by r-squared
    idxmax = LB_groups.idxmax()
    r2max = LB_groups.max()

    _,i4 = idxmax.loc[4].value
    _,i5 = idxmax.loc[5].value
    _,i6 = idxmax.loc[6].value
    
    return i4,i5,i6
    
if __name__ == '__main__':
    compute_df = True

    experiment_name = 'regselect'
    epath = os.path.join( 'results', experiment_name )
    fpath = os.path.join( epath, 'figures' )
    os.makedirs(fpath, exist_ok=True)

    num_model_instances = 100
    LB_factors = (8,7,6,5,4,3,2)
    main_csv_path = os.path.join(epath,'metrics.csv')
    if compute_df:
        t0 = time.time()
        df = run(epath,LB_factors)
        print( 'time to aggregate and organize data:',time.time()-t0 )
        df.to_csv(main_csv_path,index=False)
    else:
        df = pd.read_csv(main_csv_path)
    

    out_path = os.path.join( fpath,'performance_metrics.pdf')
    with PdfPages( out_path ) as pdf:
        ## plot regularization parameter sweep
        fig1 = plt.figure()

        ax1 = plt.subplot(311)
        i4,i5,i6 = regselect_boxplots( df, ax1, LB_factors, 'cc' )
        ax1.set_ylim((0.55,1))
        print('first reg results finish')
        ##
        ax2 = plt.subplot(312)
        testing_Bs = tuple( process_model_params( LB_factor,epath)
                             for LB_factor in LB_factors )
        # tuple( tuple( array, ... ), ... )
        lbf = len(LB_factors)
        reg1 = np.zeros((lbf,num_model_instances))
        reg2 = np.zeros((lbf,num_model_instances))        
        ratio3 = np.zeros((lbf,num_model_instances))

        for n in range(lbf):
            Bs = testing_Bs[n] # Bs is a tuple
            #
            Breg1 = np.fromiter(
                tuple( np.linalg.norm(B.reshape(-1), ord=1) for B in Bs),
                np.float32
            )
            reg1[n]=Breg1
            #
            Breg2 = np.fromiter(
                tuple( np.linalg.norm(B.reshape(-1), ord=2) for B in Bs),
                np.float32
            )
            reg2[n]=Breg2
            #
            ratio3[n] = np.fromiter(
                tuple( np.sum(np.abs(B) < 1e-3)/B.size for B in Bs ),
                np.float32
            )
        ax2.boxplot(reg1.T,
                    positions = np.arange(lbf),
                    showfliers=False,
                    widths = 0.15)
        labels = tuple('' for x in -1*np.array(LB_factors))
        ax2.set_xticks( np.arange(lbf), labels=labels )
        ax2.set_ylim(-1000,20000)    
        ax2.set_yticks((0,10000,20000),('0','1e4','2e4'))
        ax2.set_ylabel('1-norm of vec(B)')
        ##
        ax3 = plt.subplot(313)        
        ax3.boxplot(reg2.T,
                    positions = np.arange(lbf),
                    showfliers=False,
                    widths = 0.15)
        labels = tuple(f'10$^{{{x}}}$' for x in -1*np.array(LB_factors))
        ax3.set_xticks( np.arange(lbf), labels=labels)
        ax3.set_ylim(-5,100)    
        ax3.set_yticks((0,50,100))
        ax3.set_ylabel('2-norm of vec(B)')
        ax3.set_xlabel('Regularization Strength')
        
        plt.tight_layout()        
        pdf.savefig(fig1)
        plt.close(fig1)

        ##
        fig2 = plt.figure()
        (ax3L,ax3M,ax3R) = tuple( plt.subplot(3,3,i)
                                  for i in (1,2,3) )        
        # LB_factors = (8,7,6,5,4,3,2)
        # we want the set of model instances for 6,5,4
        # which corresponds to elements 2,3,4
        # of the tuple testing_Bs
        for i,j,lb,ax in zip((2,3,4),(i6,i5,i4),(6,5,4),(ax3L,ax3M,ax3R)):
            B = testing_Bs[i][j]
            ax.hist(B.reshape(-1),bins=100)
            ax.set_title(f'LB = 10$^{{{-lb}}}$')
        plt.tight_layout()        
        pdf.savefig(fig2)
        plt.close(fig2)

        print('plotting done')

