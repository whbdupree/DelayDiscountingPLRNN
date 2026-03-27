import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from observation_model_analysis import getBB
from utils import get_MAE

plt.rcParams.update({'font.family':'sans-serif'})

def run_tsne( BB, perplexity ):
    myt = TSNE(
        n_components=2,
        metric='cosine',
        perplexity=perplexity,
        random_state = 111,
    )
    BBe = myt.fit_transform(BB)
    return BBe

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', metavar='D', type=int,default=30)
    args = parser.parse_args()
    latent_size = args.latent_size
    num_instances = 100

    experiment= 'main'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{latent_size}' )

    MAEs = np.fromiter(
        (get_MAE(instance, rpath) for instance in range(num_instances)),
        np.float32
    )
    
    m = np.argsort( MAEs )[0] # best model instance
    BB , _ = getBB( rpath, latent_size )
    BBm = BB[m]
    
    out_path = os.path.join( fpath, f'cluster_exemplar_D{latent_size}.pdf' )
    with PdfPages( out_path ) as pdf:
        for perplexity in (5,30,50,100,500):
            print('Start TSNE;Kmeans ')
            t0 = time.time()
            BBme = run_tsne( BBm.T, perplexity = perplexity )
            kmeans = KMeans(n_clusters=10,n_init='auto').fit(BBme)
            print(f'run time {time.time()-t0}')
            print('End TSNE;Kmeans') 
            
            ml = kmeans.labels_
            fig = plt.figure()
            ax = plt.subplot(121)
            for i in range(10):
                idx = ml==i
                ax.plot(BBme[idx,0],BBme[idx,1],'.')
            ax.set_title(f'Perplexity = {perplexity}')
            ax.set_xlabel('tSNE1')
            ax.set_ylabel('tSNE2')
            pdf.savefig( fig )
            plt.close(fig)

