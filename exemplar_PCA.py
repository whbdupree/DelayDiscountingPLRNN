import argparse
import os
import numpy as np
import matplotlib as mpl
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial
from utils import get_MAE

rcParams.update({'font.family':'sans-serif'})

def load_data(rpath,num_instances):
    z = tuple( np.load(os.path.join( rpath, f'instance_{model_instance}_z.npz' ))['testing']
           for model_instance in range(num_instances) )
    z=np.stack(z)
    npz = np.load( os.path.join( rpath, 'zPC.npz'))
    PCs = npz['testing']
    print('loaded PCs shape:',PCs.shape)
    return PCs,z

def exemplar_PCsMN( PCs,  z, m=0,n=1 ):
    pc_shape = PCs.shape
    T = pc_shape[2]
    #print(pc_shape)
    
    fig2,((axT0,axT1),(axB0,axB1)) = plt.subplots(nrows=2,ncols=2)
    for i in range( pc_shape[0] ):
        axT0.plot(PCs[i,0,:,m],color='C0')
        axB0.plot(PCs[i,1,:,m],color='C1')
        axT1.plot(PCs[i,0,:,n],color='C0')
        axB1.plot(PCs[i,1,:,n],color='C1')
    lim_coeff = .05
    PC1range = np.array( ( np.min(PCs[...,m]),np.max(PCs[...,m]) ) )
    PC2range = np.array( ( np.min(PCs[...,n]),np.max(PCs[...,n]) ) )
    PC1d = np.diff(PC1range)
    PC2d = np.diff(PC2range)
    #print('range:',PC1range)
    #print('diff:',PC1d)
    PC1a = lim_coeff*PC1d
    PC2a = lim_coeff*PC2d     
    PC1lims = ( PC1range[0] - PC1a, PC1range[1] + PC1a )
    PC2lims = ( PC2range[0] - PC2a, PC2range[1] + PC2a )
    
    axT0.set_ylim(PC1lims)
    axT1.set_ylim(PC2lims)
    axT0.set_xticks((0,50,100,150),(-20,-10,0,10))
    axT1.set_xticks((0,50,100,150),(-20,-10,0,10))
    axT0.set_title(f'PC{m+1}')
    axT1.set_title(f'PC{n+1}')
    axT0.set_ylabel('Immediate')
    
    axB0.set_ylim(PC1lims)
    axB1.set_ylim(PC2lims)
    axB0.set_xticks((0,50,100,150),(-20,-10,0,10))
    axB1.set_xticks((0,50,100,150),(-20,-10,0,10))
    axB0.set_xlabel( 'Time (s)' )
    axB1.set_xlabel( 'Time (s)' )
    axB0.set_ylabel('Delay')

    ############### next figure
    
    fig3 = plt.figure()
    subfigs = fig3.subfigures(
        nrows = 3, ncols = 1,
        height_ratios=(0.4,0.4,0.2), 
    )
    axs = [ subfig.subplots( 1, 2, gridspec_kw={'wspace':0.3} ) 
            for subfig in subfigs[:2] ]
    ((axT2,axT3),(axB2,axB3)) = axs
    
    sf3 = subfigs[2]
    # these "blank" axes are a dirty and quick
    # solution to precisely control the position 
    # of the colorbar.
    (blank1,ax_sf3,blank2,blank3) = sf3.subplots(4,1)
    blanks = (blank1,blank2,blank3)
    [blank_axis.axis('off') for blank_axis in blanks]

    def plot_colored_segments(
            pcs,
            alt_pcs,
            dd,
            ax1,
            ax2,
            lw1 = 1.2,
            lw2 = 0.5,
    ):
        # intentionaly loop over ii thrice.
        # first, we draw the "alternate PCs" in grey.
        # then we draw the PCs of interest just plain black.
        # exactly on top of them, we draw those PCs with their
        # color-coded speed and with slightly smaller line
        # width compared to the black version of those PCs.
        # the contrast of bright color on black background
        # helps the colors pop out. finally, we plot markers
        # on top of everything to indicate the time of 
        # the choice.
        ii = pc_shape[0]
        # draw alternate PCs in grey
        for i in range( ii ):
            x_,y_ = alt_pcs[i,:,m],alt_pcs[i,:,n]
            g = 0.65
            ax1.plot(
                x_, y_,
                color = (g,)*3,
                linewidth=lw1,
            )
        # draw PCs of interest in black
        for i in range( ii ):            
            x,y = pcs[i,:,m],pcs[i,:,n]
            d=dd[i]
            ax1.plot(
                x, y,
                color = 'k',
                linewidth=lw1,
            )
            ax2.plot(
                np.arange( d.size ),
                d,
                color = 'k',
                linewidth=lw1,
            )
        # draw PCs of interest again in color
        for i in range( ii ):
            x,y = pcs[i,:,m],pcs[i,:,n]
            d = dd[i]
            for j in range( x.size-1 ):
                xx=(x[j],x[j+1])
                yy=(y[j],y[j+1])
                dj = d[j]
                ax1.plot(
                    xx, yy,
                    linewidth = lw2,
                    color=cmap(norm( dj )),
                )
            for j in range( x.size-2 ):                
                jj = (j,j+1)
                dj = d[j]
                djj = (dj,d[j+1])
                ax2.plot(
                    jj,djj,
                    linewidth = lw2,
                    color=cmap(norm( dj )),
                )
        # draw markers to indicate time of choice
        for i in range( ii ):
            x,y = pcs[i,:,m],pcs[i,:,n]
            cx,cy = x[100],y[100]
            ax1.plot( cx,cy, 'o',
                      markersize = 3,
                      markeredgecolor = 'k',
                      markerfacecolor = 'm',
            )
            # get phase point at time of choice
        ax2.plot( (100,100),(vmin,vmax),'m',alpha=0.5)
        ax2.set_yticks(np.arange(20,step=2))
    dt = 0.2
    v = np.diff(z,axis=2)/dt
    
    #print('v shape:',v.shape)

    v_ = np.sqrt( np.sum( v**2, axis=3 ) )
    v_0,v_1 = np.squeeze( np.split( v_,2,axis=1) )
    for_extrema = np.concatenate( (v_0,v_1) )
    vmax = np.max( for_extrema )
    vmin = np.min( for_extrema )
    speed_lims = (0,vmax*1.04)
    #print('speed_lims:',speed_lims)
    cmap = mpl.cm.turbo
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    plot_colored_segments( PCs[:,0],PCs[:,1], v_0, axT2, axT3 )
    plot_colored_segments( PCs[:,1],PCs[:,0], v_1, axB2, axB3 )
    
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(
        mappable,
        cax = ax_sf3,
        location = 'bottom',
        fraction = 0.08,
        label = 'Latent Factor Speed',
        ticks = np.arange(20,step=2),
    )
    ax_sf3.set_xlim((0,vmax))
    speed_lims = (0,vmax*1.04)
    axT2.set_xlim(PC1lims)
    axT2.set_ylim(PC2lims)
    axB2.set_xlim(PC1lims)
    axB2.set_ylim(PC2lims)
    axT3.set_xlim((0,150))
    axT3.set_ylim(speed_lims)
    axB3.set_xlim((0,150))    
    axB3.set_ylim(speed_lims)
    
    axT2.set_ylabel(f'PC{n+1}')        
    axT2.set_xlabel(f'PC{m+1}')
    axB2.set_ylabel(f'PC{n+1}')
    axB2.set_xlabel(f'PC{m+1}')
    
    axT3.set_ylabel(f'Latent Factor Speed')
    axT3.set_xlabel(f'Time (s)')
    axT3.set_xticks((0,50,100,150),(-20,-10,0,10))
    axB3.set_ylabel(f'Latent Factor Speed')
    axB3.set_xlabel(f'Time (s)')
    axB3.set_xticks((0,50,100,150),(-20,-10,0,10))

    subfigs[0].suptitle('Immediate Choice',fontweight='bold')
    subfigs[1].suptitle('Delay Choice',fontweight='bold')

    return fig2,fig3
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--latent_size', type=int, default=30 )
    parser.add_argument('--experiment',type=str,default='main')
    args = parser.parse_args()
    latent_size = args.latent_size
    experiment = args.experiment
    
    ncomp = 5
    num_instances = 100
    M = 10
    
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    os.makedirs( fpath ,exist_ok=True )
    rpath = os.path.join( epath, f'run{latent_size}' )
    

    MAEs = np.fromiter(
        (get_MAE(instance, rpath) for instance in range(num_instances)),
        np.float32
    )

    idx = np.argsort( MAEs )
    topM = idx[:M]

    PCs,z = load_data( rpath, num_instances )
    m = 0
    n = 1
    pdf_path = f'exemplar_latent_size{latent_size}_projections_PC{m+1}_PC{n+1}.pdf'
    my_path = os.path.join( fpath , pdf_path )
    with PdfPages( my_path ) as pdf:
        for j in range(M):
            MAE_idx = topM[j]
            print(f'model instance: {MAE_idx}')
            PCs_j = PCs[MAE_idx]
            zj = z[MAE_idx]
            fig2,fig3 = exemplar_PCsMN(
                PCs_j, zj, m=m, n=n,
            )
            suptitle = f'Model ID: {MAE_idx}'
            fig2.suptitle( suptitle )
            pdf.savefig(fig2)
            pdf.savefig(fig3)
            plt.close(fig2)
            plt.close(fig3)

