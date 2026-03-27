import glob
import numpy as np
from scipy import io
from jax import numpy as jnp
import os

data_suffixes = tuple(x[40:-4] for x in glob.glob('mean_im_de_trj/*.mat'))
data_path = 'mean_im_de_trj'

def get_spiketrains(filename,threshold = 5):
    x=io.loadmat(filename)
    data = x['spktrains_mat']
    ins = x['envinputs_mat']
    sums = [np.sum(q,axis=0) for q in data]
    if len(sums[0].shape) ==0:
        return False, None, None
    zeros = np.array([s == 0 for s in sums])
    real_spike_train = ~zeros
    idx = np.multiply(*real_spike_train)
    sum_idx = np.sum(idx)
    if sum_idx < threshold:
        return False, None, None
    clean_data = data[:,:,idx]
    return True, sum_idx, clean_data

def data_loader():
    obs = {}
    for data_suffix in data_suffixes:
        file_mat = f'spktrains_mean_by_choice_{data_suffix}'
        flag, ncells, session_obs = get_spiketrains(
            os.path.join(data_path, file_mat)        
        )
        if flag:
            obs[data_suffix] = jnp.array( session_obs ) # <- jax array
    return obs

def get_MAE(instance, rpath, group='testing'):
    npz_file = f'instance_{instance}_error.npz'
    load_me = os.path.join(rpath,npz_file)
    npz = np.load( load_me )
    MAE = npz[group]
    return MAE[-1]

