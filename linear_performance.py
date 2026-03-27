import os
import time
import argparse
import pickle
import pprint
import numpy as np
import pandas as pd
from jax import tree,vmap
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from functools import partial
from utils import get_MAE

plt.rcParams.update({
    'font.family':'sans-serif',
})

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

get_item = lambda a:a.item()
loader = lambda a: tree.map( get_item, dict(np.load(a,allow_pickle=True)) )

def process_metric_trees(mnpz, model_instance, latent_size ):
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
                      'latent_size':latent_size,
                      'recording':rkey,
                      'value':value}
                )

    return pd.DataFrame(result)

def process_model_instance( model_instance, latent_size, epath ):
    rpath = os.path.join( epath, f'run{latent_size}' )
    mpath = os.path.join(
        rpath,
        f'instance_{model_instance}_metrics.npz'
    )
    mnpz = np.load( mpath, allow_pickle=True )
    
    df = process_metric_trees(mnpz,model_instance,latent_size)
    
    return df

def run(epath,latent_sizes):
    num_model_instances = 100
    df = pd.concat(
        tuple( process_model_instance( model_instance, latent_size, epath)
               for latent_size in latent_sizes
               for model_instance in range(num_model_instances) )
    )
    return df

def main_boxplots(df,axs,metric='cc'):
    ax1,ax2 = axs
    df = df[df['metric']==metric]    
    data = df.groupby(['group','latent_size'])
    
    loc = np.array((-0.2,0,0.2,0.2))
    colors= ('r','b','g','y','c')
    latent_sizes = (3,5,10,20,50,100,200,500,1000)
    flierprops=dict(marker='x', markersize=3)    
    groups = ('training','testing')
    for g,group in enumerate(groups):
        group_data = tuple( np.array(data.get_group((group,latent_size)).groupby('model_instance')['value'].mean())
                            for latent_size in latent_sizes )
        pos = loc[g]
        color=colors[g]
        l = len(group_data)
        hh=ax1.boxplot( group_data,
                        patch_artist = True,
                        positions = np.arange(l)+pos,
                        showfliers=False,
                        widths = 0.15)
        color_boxplots(hh,color)
    ax1.set_xticks( np.arange(len(latent_sizes)), latent_sizes )
    ax1.set_ylabel('Model Instance CC')
    groups = ('testing','scramble','shuffle5','shuffle15')
    group_dict = {}
    for g,group in enumerate(groups):
        group_data = tuple( np.array(data.get_group((group,latent_size)).groupby('model_instance')['value'].mean())
                            for latent_size in latent_sizes )
        group_dict[group] = group_data
        pos = loc[g]
        color=colors[1:][g] # skip training color
        l = len(group_data)
        hh=ax2.boxplot( group_data,
                        patch_artist = True,
                        positions = np.arange(l)+pos,
                        showfliers=False,
                        widths = 0.15)
        color_boxplots(hh,color)
    print('scramble:')
    zip1 = zip( group_dict['testing'],group_dict['scramble'],latent_sizes )
    pprint.pprint(
        { z:ttest_ind(x,y)
          for x,y,z in zip1 }
    )
    print('shuffle5:')
    zip2 = zip( group_dict['testing'],group_dict['shuffle5'],latent_sizes )
    pprint.pprint(
        { z:ttest_ind(x,y)
          for x,y,z in zip2 }
    )
    print('shuffle15:')
    zip3 = zip( group_dict['testing'],group_dict['shuffle15'],latent_sizes )
    pprint.pprint(
        { z:ttest_ind(x,y)
          for x,y,z in zip3 }
    )
    
    ax2.set_xticks( np.arange(len(latent_sizes)), latent_sizes )
    ax2.set_ylabel('Model Instance CC')
    ax2.set_xlabel('No. Variables in Latent Model')

def model_selection_boxplots(df,ax, metric='cc'):
    df = df[df['metric']==metric]    
    data = df.groupby(['group','latent_size'])
    loc = np.array((-0.2,0,0.2,0.2))
    colors= ('r','b','g','y','c')
    latent_sizes = (10,20,30,40,50)
    flierprops=dict(marker='x', markersize=3)    
    groups = ('training','testing')
    for g,group in enumerate(groups):
        group_data = tuple( np.array(data.get_group((group,latent_size)).groupby('model_instance')['value'].mean())
                            for latent_size in latent_sizes )
        pos = loc[g]
        color=colors[g]
        l = len(group_data)
        hh=ax.boxplot( group_data,
                       patch_artist = True,
                       positions = np.arange(l)+pos,
                       showfliers=False,
                       widths = 0.15)
        color_boxplots(hh,color)
    ax.set_xticks( np.arange(len(latent_sizes)), latent_sizes )
    ax.set_ylabel('Model Instance CC')
    ax.set_xlabel('No. Variables in Latent Model')
    
if __name__ == '__main__':
    compute_df = True
    experiment_name = 'linear'
    epath = os.path.join( 'results', experiment_name )
    fpath = os.path.join( epath, 'figures' )
    os.makedirs(fpath, exist_ok=True)

    ## main performance plots
    latent_sizes = (10,20,30,40,50)
    main_csv_path = os.path.join(epath,'metrics.csv')
    if compute_df:
        t0 = time.time()
        df = run(epath,latent_sizes)
        print( 'time to aggregate and organize data:',time.time()-t0 )
        df.to_csv(main_csv_path,index=False)
    else:
        df = pd.read_csv(main_csv_path)
    
    out_path = os.path.join( fpath,'performance_metrics.pdf')
    with PdfPages( out_path ) as pdf:
        ## plot model selection results with useful latent sizes
        fig = plt.figure()
        ax = plt.subplot(211)
        model_selection_boxplots(df,ax)
        pdf.savefig(fig)

