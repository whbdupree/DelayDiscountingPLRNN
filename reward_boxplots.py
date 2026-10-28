import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import pprint

def plot_reward_boxplots(total_rewards, imm_rewards, impulsivity, out_path):
    """Generate and save ordered reward boxplots colored by impulsivity metrics.
       The figure is written directly to disk at `out_path` and closed.
    """

    
    def panel2(data, ax, y, c):
        cmap = plt.get_cmap("GnBu") 
        norm = mcolors.Normalize(vmin=np.min(y), vmax=np.max(y))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        #
        xticklabels = [""] * len(data)
        medianprops={'color':'m'}
        widths = 0.6
        bp = ax.boxplot(
            data,
            tick_labels=xticklabels,
            medianprops=medianprops,
            widths=widths,
            patch_artist=True,
        )
        #
        colors = sm.to_rgba(y)
        for color, patch in zip(colors, bp["boxes"]):
            patch.set_facecolor(color)
        ax.set_xlabel("Recording Session")
        ax.set_ylabel("Immediate Trial Pellets")
        #
        cbar=fig.colorbar(sm, ax=ax)
        cbar.ax.set_yticks((0.4,0.5,0.6,0.7,0.8),("","0.5","","0.7",""))
        cbar.ax.set_ylabel("Delay Choice Fraction")
        #
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xticks( ax.get_xticks(),xticklabels )
        xticklines = ax_top.get_xticklines()
        for i, xtickcolor in enumerate(c):
            xticklines[i * 2].set_markeredgecolor(xtickcolor)
            xticklines[i * 2 + 1].set_markeredgecolor(xtickcolor)
        ax_top.xaxis.set_tick_params(width=4)
        return ax.get_xlim()
    
    def panel1(data,ax,y,c,xlim):
        cmap = plt.get_cmap("GnBu") 
        norm = mcolors.Normalize(vmin=np.min(y), vmax=np.max(y))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        #

        #for yi,datum in enumerate( zip(data,y):
        for i in range(len(data)):
            datum = data[i]
            yi=y[i] 
            ax.plot(
                i,datum,'.',
                markerfacecolor=cmap(norm(yi)),
                markeredgecolor='k',
                markersize=15,
            )
        #
        ld = len(data)
        xticklabels = ('',)*ld
        ax.set_xticks(range(ld),xticklabels)
        ax.set_xlabel("Recording Session")
        ax.set_ylabel("Total Session Pellets")
        ax.set_xlim(xlim)
        #
        cbar=fig.colorbar(sm, ax=ax)
        cbar.ax.set_yticks((0.4,0.5,0.6,0.7,0.8),("","0.5","","0.7",""))
        cbar.ax.set_ylabel("Delay Choice Fraction")
        #
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xticks( ax.get_xticks(),xticklabels )
        xticklines = ax_top.get_xticklines()
        for i, xtickcolor in enumerate(c):
            xticklines[i * 2].set_markeredgecolor(xtickcolor)
            xticklines[i * 2 + 1].set_markeredgecolor(xtickcolor)
        ax_top.xaxis.set_tick_params(width=4)
        

    delayT = impulsivity["delayT"].to_numpy()
    top_tick_colors = np.array( tuple( 'k' if x == 8 else 'w' for x in delayT ) )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")

    # ax2; immediate trial reward distributions
    labels = list(imm_rewards.keys())    
    data = [imm_rewards[k] for k in labels]    
    medians = [np.median(x) for x in data]
    order = np.argsort(medians)
    y = impulsivity["delay_choice_percent"].to_numpy()
    ordered_data = [data[i] for i in order]
    xlim=panel2( ordered_data, ax2, y[order], top_tick_colors[order])
    
    # ax1; total session reward
    labels = list(total_rewards.keys())
    data = [total_rewards[k] for k in labels]
    order = np.argsort(data)
    ordered_data = [data[i] for i in order]
    panel1( ordered_data, ax1, y[order], top_tick_colors[order],xlim) 
    fig.savefig(out_path)
    plt.close(fig)

def main(input_file='mean_im_de_trj/processed_events.csv', output_dir='results/figures'):
    os.makedirs(output_dir, exist_ok=True)

    
    ## get dataframe with valid sessions
    valid_sessions = pd.read_csv('results/main/metrics.csv')['recording'].unique()
    df = pd.read_csv(input_file)
    df = df[df['session_name'].isin(valid_sessions)]    

    filtered_df = df.loc[(~df['isdelay'])]
    imm_rewards_by_session = {
        session_name: sub_df['pellets_per_trial'].values
        for session_name, sub_df in filtered_df.groupby('session_name')
    }
    total_rewards_by_session=df.groupby('session_name')['pellets_per_trial'].sum() 

        

    impulsivity = dict( np.load('mean_im_de_trj/impulsivity.npz',allow_pickle=True) )
    imp_by_session = {
        session_name: impulsivity[session_name].item()
        for session_name, sub_df in filtered_df.groupby('session_name')
    }
    #pprint.pprint(imp_by_session)
    imp=pd.DataFrame.from_dict( imp_by_session,orient='index' )
    #pprint.pprint(imp)
    reward_pdf_path = os.path.join(output_dir, 'reward_boxplots.pdf')
    plot_reward_boxplots(total_rewards_by_session, imm_rewards_by_session, imp, reward_pdf_path)


if __name__ == '__main__':
    main()
