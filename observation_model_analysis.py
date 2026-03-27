import pickle
import os
from jax import tree
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
nums_clusters=[ 5,10,20 ]

def run_kmeans(model_instance,BB):
    #print(f'Start Kmeans model_instance {model_instance}')
    labels_list = []
    col_names = []
    for i,num_clusters in enumerate(nums_clusters):
        kmeans = KMeans(n_clusters=num_clusters,n_init='auto').fit(BB)
        ml = kmeans.labels_
        labels_list.append(ml)
    col_names = [f'nc{nc}'for nc in nums_clusters]
    col_names.extend(['model_instance']) # list of four strings
    lll = BB.shape[0]
    m_list = [model_instance]*lll
    col_vals = labels_list + [m_list]
    df = pd.DataFrame(
        dict(zip(col_names, col_vals))
    )
    df.index.name = 'nID'
    return df

def getBB(rpath,D):
    prototype_obs_path = os.path.join( rpath, 'instance_0_obs.npz' )
    prototype_obs_npz = np.load( prototype_obs_path, allow_pickle=True )
    obs_training = prototype_obs_npz['training'].item()
    obs_testing = prototype_obs_npz['testing'].item()
    training_keys = tuple( obs_training.keys() )
    testing_keys = tuple( obs_testing.keys() )
    obs_keys = training_keys + testing_keys
    obs = {**obs_training, **obs_testing}
    sorted_lengths = np.array( tree.leaves( tree.map( lambda a:a.shape[-1], obs) ) )
    num_instances = 100
    BB = np.zeros((num_instances,D,2563))
    for model_instance in range(num_instances):
        params_path = os.path.join(rpath,f'params_{model_instance}.pkl')
        with open( params_path,'rb') as file:
            params = pickle.load(file)
        B_tree_training = params['training']['B_tree']
        B_tree_testing = params['testing']['B_tree']
        B_tree = {**B_tree_training, **B_tree_testing}
        B = tuple( B_tree[key].kernel.value for key in obs_keys )
        B = np.concat(B,axis=-1)
        BB[model_instance] = B
    return BB, sorted_lengths
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', metavar='D', type=int,default=30)
    args = parser.parse_args()
    
    D = args.latent_size

    experiment= 'main'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{D}' )

    BB, sorted_lengths = getBB( rpath,D )

    _df = []
    for m,Bm in enumerate(BB):
        _df.append(  run_kmeans( m, Bm.T ) )
    df = pd.concat( _df )
    out_path = os.path.join( rpath, f'loading_matrix_clusters.csv' )
    df.to_csv( out_path )

    sl_path = os.path.join( rpath, 'sorted_lengths.npy' )
    with open( sl_path, 'wb') as file:
        np.save( file, sorted_lengths )
