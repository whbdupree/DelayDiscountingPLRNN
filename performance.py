import os
import time
import pickle
import pprint
import numpy as np
import pandas as pd
from jax import tree,vmap
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils import get_MAE

plt.rcParams.update({
    'font.family':'sans-serif',
})

def bootstrap(data1, data2, N=10000):
    M = data1.size # 100 model instances
    
    bs_index1 = np.random.randint(M, size=(N, M))
    bs_data1 = data1[bs_index1]
    q1_data1 = np.quantile( bs_data1,0.25,axis=1)
    
    bs_index2 = np.random.randint(M, size=(N, M))
    bs_data2 = data2[bs_index2]
    q3_data2 = np.quantile( bs_data2,0.75,axis=1)
    
    test = q3_data2 > q1_data1
    counts = np.sum(test)
    p_value = counts / N
    
    return p_value

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

def process_metric_trees(mnpz, mpkl, model_instance, latent_size ):
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
    #
    # shuffle_results_foo.pkl
    # shuffle5: ccs: recording: array
    # shuffle5: r2s: recording: array
    # shufle15 ...
    
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
    m_name_map={'ccs':'cc','r2s':'r2'}
    for key in mpkl.keys():
        group = mpkl[key]
        for metric in ('ccs','r2s'):
            metric_tree = group[metric]
            for rkey in metric_tree.keys():
                data = metric_tree[rkey]
                value = np.mean(data).item()
                result.append(
                    { 'group':key,
                      'metric':m_name_map[metric],
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
    mpath = os.path.join(
        rpath,
        f'shuffle_results_{model_instance}.pkl'
    )
    mpkl = pickle.load( open( mpath, 'rb' ) )
    
    df = process_metric_trees(mnpz,mpkl,model_instance,latent_size)
    
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
        { z:bootstrap(x,y)
          for x,y,z in zip1 }
    )
    print('shuffle5:')
    zip2 = zip( group_dict['testing'],group_dict['shuffle5'],latent_sizes )
    pprint.pprint(
        { z:bootstrap(x,y)
          for x,y,z in zip2 }
    )
    print('shuffle15:')
    zip3 = zip( group_dict['testing'],group_dict['shuffle15'],latent_sizes )
    pprint.pprint(
        { z:bootstrap(x,y)
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
    experiment_name = 'main'
    epath = os.path.join( 'results', experiment_name )
    fpath = os.path.join( epath, 'figures' )
    os.makedirs(fpath, exist_ok=True)

    ## main performance plots
    latent_sizes = (3,5,10,20,30,40,50,100,200,500,1000)        
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
        ## plot main results with broad range of latent sizes
        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)    
        main_boxplots( df, (ax1,ax2) )
        pdf.savefig(fig)

        ## plot model selection results with useful latent sizes
        fig = plt.figure()
        ax = plt.subplot(211)
        model_selection_boxplots(df,ax)
        pdf.savefig(fig)

    ## specific performance plots for best model by MAE in latent_size = 30
    latent_size = 30
    num_instances = 100
    rpath = os.path.join( epath, f'run{latent_size}' )
    MAEs = np.fromiter(
        (get_MAE(instance, rpath) for instance in range(num_instances)),
        np.float32
    )
    top_model = np.argsort( MAEs )[0]
    ## get MAE by neuron
    obs_path = os.path.join( rpath,f'instance_{top_model}_obs.npz')
    obs_npz = np.load(obs_path,allow_pickle = True)
    obs_tree = { file:obs_npz[file].item() for file in obs_npz.files }
    #
    x_path = os.path.join( rpath,f'instance_{top_model}_x.npz')
    x_npz = np.load(x_path,allow_pickle = True)
    x_tree = { file:x_npz[file].item() for file in x_npz.files }
    #
    f = lambda a,b: np.mean(np.abs(a-b),axis=(0,1))
    MAE_by_neuron = tree.map( f, obs_tree, x_tree )
    ## get CC by neuron
    metrics_path = os.path.join( rpath,f'instance_{top_model}_metrics.npz')
    m_npz = np.load( metrics_path, allow_pickle = True )
    CC_tree = { file:m_npz[file][0] for file in m_npz.files }
    #
    g = lambda a: np.mean(a,axis=0)
    CC_by_neuron = tree.map(g,CC_tree)
    #
    z_path = os.path.join( rpath,f'instance_{top_model}_z.npz')
    z = np.load( z_path )['testing'] # this will yield an array
    
    ## plot MAE,CC
    def my_boxplots(
            data,
            ax,
            ylabel,
            yticks,
            xticks = {'ticks':(1,1.5),'labels':('Training','Testing')},
            w = 0.4,
            xlim = (0.7,1.8),
            colors = ('r','b')
    ):
        h0 = ax.boxplot(
            data[0],
            positions = (1,),
            bootstrap = 10000,
            widths = w,
            notch = True,
            showfliers = False,
            patch_artist = True
        )
        h1 = ax.boxplot(
            data[1],
            positions = (1.5,),
            bootstrap = 10000,
            widths = w,
            notch = True,
            showfliers = False,
            patch_artist = True
        )
        [color_boxplots( h, c ) for h,c in zip((h0,h1),colors)]
        ax.set_xlim(xlim)
        ax.set_xticks(xticks['ticks'],xticks['labels'])
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel)

    def plot_z( z_,b,n0):
        _,_,T,_ = z_.shape
        z = z_[b]
        t = 0.2*np.arange(T)-20
        fig,axs = plt.subplots(3,2)
        n = n0
        for i in range(3):
            for j in range(2):
                ax=axs[i][j]
                ax.plot(t,z[0,:,n],'C0')
                ax.plot(t,z[1,:,n],'C1')
                n=n+1
        return fig,axs

    def plot_x_obs( x_,obs_,b):
        key = tuple(x_.keys())[b]
        x = x_[key]
        o = obs_[key]
        _,T,_ = x.shape
        t = 0.2*np.arange(T) -20
        fig,axs = plt.subplots(3,2)
        n = 0
        for i in range(3):
            for j in range(2):
                ax=axs[i][j]
                ax.plot(t,o[0,:,n],'C0')
                ax.plot(t,o[1,:,n],'C1')
                ax.plot(t,x[0,:,n],'--C0')
                ax.plot(t,x[1,:,n],'--C1')
                n=n+1
        return fig,axs
    out_path = os.path.join( fpath,f'exemplar_model_instance_performance.pdf')
    with PdfPages(out_path) as pdf:
        ## individual model boxplots
        fig1 = plt.figure()        
        #
        ax1 = plt.subplot(131)
        mae = ( np.concatenate(tree.leaves(MAE_by_neuron['training'])),
                np.concatenate(tree.leaves(MAE_by_neuron['testing'])) )
        my_boxplots( mae,ax1,
                     'Mean Absolute Error by Neuron',
                     (0,0.05,0.1,0.15,0.2) )
        #
        ax2 = plt.subplot(133)
        cc = ( np.concatenate(tree.leaves(CC_by_neuron['training'])),
               np.concatenate(tree.leaves(CC_by_neuron['testing'])) )
        my_boxplots( cc, ax2,
                     'Mean Correlation by Neuron',
                     (0.7,0.8,0.9,1))
        #
        pdf.savefig( fig1 )
        plt.close( fig1 )

        ## individual model z traces
        b = 0
        n0 = 24
        fig2,axs = plot_z( z,b,n0)
        pdf.savefig(fig2)
        plt.close(fig2)
        
        ## individual model obs,x traces
        x = x_tree['testing']
        obs = obs_tree['testing']
        fig3,axs = plot_x_obs(x,obs,b)
        pdf.savefig(fig3)
        plt.close(fig3)

