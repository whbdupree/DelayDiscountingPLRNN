import numpy as np
from jax import nn,jit,vmap,numpy as jnp
import argparse
import os
import pickle
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn import metrics

plt.rcParams.update({
    'font.family':'sans-serif',
})

def sanity(z,z0_,A,W,h):
    def f(p,z):
        A,W,h = p
        return A*z + nn.relu(z)@W + h
    batches,contexts,T,latent_size = z.shape
    z0 = z0_[0,0]
    zT = np.zeros((T,latent_size))
    zT[0] = z0
    for t in range(T-1):
        zT[t+1] = f( (np.diag(A),W,h), zT[t] )
    return z[0,0],zT

def getJ(A,W,z):
    zg = z>0
    DO = np.diag(zg)
    return A+DO@W, zg

def compute_eigs( z, model, model_instance, threshold=1e-5 ):
    latent_model = model.latent_model
    z0 = model.z0.value
    A = np.diag( latent_model.A.value )

    Wh = latent_model.Wh
    W = Wh.kernel.value
    h = Wh.bias.value
    
    batches,contexts,T,latent_size = z.shape
    icount = np.zeros((batches,contexts,T))
    real_eigs = np.zeros((batches,contexts,T,latent_size))
    imag_eigs = np.zeros((batches,contexts,T,latent_size))
    states = np.zeros((batches,contexts,T,latent_size))
    for i in range(batches):
        for j in range( contexts ):
            for t in range(T):
                zijt = z[i,j,t]
                J,zg = getJ(A,W,zijt)
                states[i,j,t] = zg
                vals,vecs = np.linalg.eig( J )
                ei = vals.imag
                er = vals.real
                critical = np.abs(er) > 1e-5
                num_imaginary = np.nonzero(ei[critical])[0].size
                icount[i,j,t] = num_imaginary
                real_eigs[i,j,t] = er
                imag_eigs[i,j,t] = ei
    return (A,W,h,z0), icount, real_eigs, imag_eigs, states

def make_figs( z, model, model_instance ):

    batches,contexts,T,latent_size = z.shape
    params, icount, real_eigs, imag_eigs, states = compute_eigs(
        z, model, model_instance )
    
    fig = plt.figure()
    axT = plt.subplot(211)
    axB = plt.subplot(212)
    axs = (axT,axB)
    colors = ('C0','C1')

    for i in range(batches):
        for j in range( contexts ):
            ax = axs[j]
            ax.plot( icount[i,j],color = colors[j],alpha=0.5 )
    # plot mean icounts on top
    mi = np.mean( icount, axis=0 )
    axT.plot(mi[0],'k',linewidth=2)
    axB.plot(mi[1],'k',linewidth=2)
    maxi = np.max(icount)
    mini = np.min(icount)
    diffi = maxi - mini
    ylims = ( mini - .05*diffi, maxi + 0.05*diffi  )
    axT.set_ylim(ylims)
    axB.set_ylim(ylims)
    xlims = (0,150)
    axT.set_xlim(xlims)
    axB.set_xlim(xlims)    
    
    axT.set_title('Imaginary Eigenvalue Count')
    axT.set_ylabel('Immediate')
    axB.set_ylabel('Delay')
    axB.set_xlabel('Time (s)')
    axT.set_xticks((0,50,100,150),("","","",""))
    axB.set_xticks((0,50,100,150),(-20,-10,0,10))
    
    fig.suptitle(f'Model Instance: {model_instance}')
    fig.tight_layout()

    ## plot real components in time
    fig2 = plt.figure()
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412)
    i=0
    for d in range(latent_size):
        ax1.plot( real_eigs[i,0,:,d],color='C0',alpha=0.5 )
        ax2.plot( real_eigs[i,1,:,d],color='C1',alpha=0.5 )
    
    ax1.set_xticks((0,50,100,150),("","","",""))
    ax2.set_xticks((0,50,100,150),(-20,-10,0,10))
    ax1.set_title('Real Eigenvalue Components')
    
    ## plot eigenspectrum
    def spectrum(ax,ri,lims):
        reb,ieb = ri
        xlim,ylim = lims        
        mks = 3
        #
        def plot(ax,re,ie,m,c):
            ax.plot( re, ie, m,
                     color = c,
                     alpha = 0.75 )
        plot( ax, reb[0], ieb[0], '.', 'C0')
        plot( ax, reb[1], ieb[1], 'x', 'C1' )
        #
        q = np.linspace(0,2*np.pi,456)                
        ax.plot( np.cos(q),np.sin(q),color=[0.5,0.5,0.5] )
        #
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Re. λ')
        ax.set_ylabel('Im. λ')
        
    figE = plt.figure()
    axTL = plt.subplot(221)
    axTL.set_aspect('equal')
    axTL.set_title('Eigenvalues')
    axTR = plt.subplot(222)
    axTR.set_aspect('equal')    
    axTR.set_title('Eigenvalues')
    batch=0
    ts=(50,100)
    reigs = real_eigs[batch,:,ts]
    ieigs = imag_eigs[batch,:,ts]
    def scale( e, w=0.2 ):
        m = ( np.min(e), np.max(e) )
        s = ( m[1]-m[0] ) * w
        return ( m[0]-s, m[1]+s )
    xlim = scale( reigs )
    ylim = scale( ieigs )
    lims = ( xlim, ylim )
    #t = 50
    spectrum( axTL,
              ( reigs[:,0], ieigs[:,0] ),
              lims )
    #t = 100
    spectrum( axTR,
              ( reigs[:,1], ieigs[:,1] ),
              lims )
    plt.tight_layout()

    return fig,fig2,figE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int, default = 30)
    parser.add_argument('--read_counts', action='store_true', default=False)
    args = parser.parse_args()
    D = args.dim_z
    read_counts = args.read_counts

    experiment = "main"
    epath = os.path.join( os.getcwd(), 'results',experiment )
    rpath = os.path.join( epath, f'run{D}' )
    fpath = os.path.join( epath, 'figures' )
    
    ## get the 10 best models by MAE
    MAEs = np.zeros(100)
    for model_instance in range(100):
        error_path = os.path.join( rpath, f"instance_{model_instance}_error.npz")
        error = np.load( error_path, allow_pickle = True )["testing"][-1]
        MAEs[model_instance] = error
    top10 = np.argsort(MAEs)[:10]
    
    
    batches = 42
    T = 151
    contexts = 2
    latent_size = D
    
    counts_out = os.path.join( rpath, 'eigen_counts.npz' )
    if read_counts:
        q = np.load( counts_out )
        all_icount = q['all_icount']
        min_icount = q['min_icount']
        all_unique_states = q['all_unique']
        min_unique_states = q['min_unique']
    else:
        min_unique_states = np.zeros(100)
        all_unique_states = np.zeros((100,batches,contexts))
        min_icount = np.zeros(100)
        all_icount = np.zeros((100,batches,contexts,T))
        for model_instance in range(100):
            data_path = os.path.join(
                rpath, f'instance_{model_instance}_z.npz' )
            znpz = np.load( data_path )
            z = znpz['testing']
            params_path = os.path.join(
                rpath, f'params_{model_instance}.pkl' )
            with open( params_path, 'rb') as file:
                params = pickle.load( file )
            model = params['testing']
            _, icount, _, _,states= compute_eigs(
                z, model, model_instance )
            min_icount[model_instance]=np.min(icount)
            all_icount[model_instance] = icount
            unique_states = np.zeros( (batches,contexts) )
            for b in range(batches):
                for c in range(contexts):
                    uniq = np.unique( states[b,c], axis=0 )
                    num_unique,_ = uniq.shape
                    # number of unique states traversed by one trajectory
                    unique_states[b,c] = num_unique
            all_unique_states[model_instance] = unique_states
            min_unique_states[model_instance] = np.min( unique_states )
        np.savez(
            counts_out,
            **{ 'all_icount':all_icount,
                'min_icount':min_icount,
                'all_unique':all_unique_states,
                'min_unique':min_unique_states }
        )
        

    out_path = os.path.join( fpath,'imaginary_components.pdf')
    with PdfPages( out_path ) as pdf:
        fig0a = plt.figure()
        ax = plt.subplot(221)
        #ax.set_aspect('equal')
        ax.hist(min_unique_states,bins=20,color='k')
        ax.boxplot(
            min_unique_states,
            orientation='horizontal',
            positions=(-2,),
            widths=2,
            notch=True,
            bootstrap=10000,
            showfliers=False,
            manage_ticks=False,
        )
        xlim = ax.get_xlim()
        ax.plot(xlim,(0,0),'--',color=(0.5,0.5,0.5))
        ax.set_xlim(xlim)
        ax.set_ylabel('Model Instance Minimum\nUnique Linear Subregions')
        pdf.savefig(fig0a)
        no_i = np.sum( all_icount == 0 )
        print( '' )
        print( f'states with strictly real eigenvalues: {no_i}' )
        print( f'which is {100*no_i/all_icount.size} % of all states')
        print( f'model instances that contain states with strictly real eigenvalues: {np.sum( min_icount == 0 )}' )
        print( f'those model instances are:')
        print( np.where(min_icount==0)[0] )
        print( '' )
        print( f'model instance median( minimum unique states ): {np.median(min_unique_states)}' )
        print( '' )

        
        print('figures for top 10 model instances:')
        for model_instance in top10:
            print('model_instance:',model_instance)
            data_path = os.path.join(
                rpath, f'instance_{model_instance}_z.npz' )
            znpz = np.load( data_path )
            z = znpz['testing']
            zi = z[:,0]
            zd = z[:,1]
            
            params_path = os.path.join(
                rpath, f'params_{model_instance}.pkl' )
            with open( params_path, 'rb') as file:
                params = pickle.load( file )
            model = params['testing']
            eig_figs = make_figs(z,model,model_instance)
            fig,fig2,figE = eig_figs
            pdf.savefig(fig)
            pdf.savefig(fig2)
            pdf.savefig(figE)
            plt.close(fig)            
            plt.close(fig2)
            plt.close(figE)            


