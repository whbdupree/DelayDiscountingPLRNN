import os
import optax
import yaml
import argparse
import pickle
import numpy as np
from jax import numpy as jnp
from jax import tree, vmap, random
from flax import nnx
from scipy import integrate, stats, ndimage, io
from functools import partial
from operator import itemgetter

from nnx_plrnn import Net
from utils import data_loader
from optimize import compute_r2,compute_loss,run_model

def ml_passes(
    rngs,
    obs,
    train_epochs,
    test_epochs,
    #
    latent_size,
    hyperparameters
):

    T,contexts,L_Breg, l1_ratio = hyperparameters

    len_obs = len(obs)
    train_fraction = 10
    test_fraction = len_obs - train_fraction
    r = random.permutation( rngs(), jnp.arange( len_obs ) )

    obs_dictionary_keys = tuple(obs.keys())

    obs_keys_train = tuple(
        obs_dictionary_keys[ rt ] for rt in r[:train_fraction]
    )
    obs_keys_test  = tuple(
        obs_dictionary_keys[ rt ] for rt in r[train_fraction:]
    )
    
    obs_train = dict(zip(
        obs_keys_train,
        itemgetter( *obs_keys_train )(obs)
    ))
    obs_test = dict(zip(
        obs_keys_test,
        itemgetter( *obs_keys_test )(obs)
    ))

    my_size = lambda a: a.shape[-1]
    session_sizes_train = tree.map( my_size, obs_train)
    session_sizes_test = tree.map( my_size, obs_test)

    model_train = Net(rngs, session_sizes_train, latent_size, contexts, T)
    model_test = Net(rngs, session_sizes_test, latent_size, contexts, T)
    model_SW = Net(rngs, session_sizes_test, latent_size, contexts, T)    

    loss_partial = partial(
        compute_loss,
        L_Breg=L_Breg,
        l1_ratio=l1_ratio,
    )
    schedule_train = optax.warmup_cosine_decay_schedule(
        # schedule_train = schedule_jit(
        init_value=0,
        peak_value=4e-3,
        warmup_steps=0,
        decay_steps=train_epochs,
        end_value=5e-4,
    )
    schedule_test = optax.warmup_cosine_decay_schedule(
        # schedule_test = schedule_jit(
        init_value=0,
        peak_value=4e-3,
        warmup_steps=0,
        decay_steps=test_epochs,
        end_value=5e-4,
    )

    # DEFINE OPTIMIZERS

    train_params = nnx.All(nnx.Param)
    diff_state_train = nnx.DiffState(0, train_params)


    #optimizer_train = nnx.Optimizer(
    optimizer_train = nnx.ModelAndOptimizer(
        model_train,
        tx=optax.chain(optax.clip(0.2), optax.adam(learning_rate=1e-3)),
        wrt=train_params,
    )

    get_ccs = vmap(vmap(
        lambda a, b: jnp.corrcoef(a,b)[0, 1],
        in_axes=1
    ))
    get_r2s = vmap(vmap(
        lambda a, b: compute_r2(a,b),
        in_axes=1
    ))
    x_train, z_train, loss_train, MAEs_train, params_train = run_model(
        optimizer_train, loss_partial, train_epochs, obs_train, diff_state_train, train_bool=True
    )

    ccs_train = tree.map(get_ccs, obs_train, x_train)
    r2s_train = tree.map(get_r2s, obs_train, x_train)
    # #######TESTING#######
    
    test_params = nnx.Not(nnx.PathContains('latent_model'))    
    diff_state_test = nnx.DiffState(0, test_params)

    # # overwrite
    model_test.latent_model = model_train.latent_model

    #optimizer_test = nnx.Optimizer(
    optimizer_test = nnx.ModelAndOptimizer(
        model_test,
        tx=optax.chain(optax.clip(0.2), optax.adam(learning_rate=1e-3)),
        wrt=test_params,
    )
    
    # run
    x_test, z_test, loss_test, MAEs_test, params_test = run_model(
        optimizer_test, loss_partial, test_epochs, obs_test, diff_state_test, train_bool=False
    )

    ccs_test = tree.map(get_ccs, obs_test, x_test)
    r2s_test = tree.map(get_r2s, obs_test, x_test)
    
    train_result = (
        x_train,
        obs_train,
        loss_train,
        (ccs_train,r2s_train),
        MAEs_train,
        z_train,
        model_train,
    )
    test_result = (
        x_test,
        obs_test,
        loss_test,
        (ccs_test,r2s_test),
        MAEs_test,
        z_test,
        model_test,
    )
    return train_result, test_result, obs_keys_train, obs_keys_test


def main(experiment_name,Net,model_instance,latent_size,LB_factor,key):
    rngs = nnx.Rngs(key)
    
    l1_ratio = 0.5
    L_Breg = 10**LB_factor

    # output filenampe

    
    epath = os.path.join(os.getcwd(), 'results', experiment_name)
    
    os.makedirs(epath, exist_ok=True)
    rpath_relative = ('results',experiment_name,f'run{int(abs(LB_factor))}')
    rpath = os.path.join(os.getcwd(),*rpath_relative)
    os.makedirs(rpath, exist_ok=True)

    train_epochs = 7000
    test_epochs = 3000
    # model hyperparameters

    T = 151
    contexts = 2

    # create config yaml
    config_dict = {
        'train_epochs': train_epochs,
        'test_epochs': test_epochs,
        'latent_factors': latent_size,
        'l1_ratio': l1_ratio,
        'L_Breg': L_Breg,
    }

    filestr = os.path.join(rpath, f'instance_{model_instance}_config.yaml')
    with open(filestr, 'w') as file:
        yaml.dump(config_dict, file)

    obs = data_loader()

    hyperparameters = ( T,contexts,L_Breg, l1_ratio )
    
    train_result, test_result,train_idxs, test_idxs = ml_passes(
        rngs,
        obs,
        train_epochs,
        test_epochs,
        latent_size,
        hyperparameters
    )

    print( 'model size:',latent_size)
    print( 'model instance:',model_instance)
    catCC = jnp.concatenate(tree.leaves(train_result[3][0]), axis=1)
    print( 'mean training CC:', np.mean( catCC) )
    catCC = jnp.concatenate(tree.leaves(test_result[3][0]), axis=1)
    print( 'mean testing CC:', np.mean( catCC) )

                        
    # MODEL RESULT FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_x.npz'),
        **{'training': train_result[0], 'testing': test_result[0]},
    )

    # OBSERVATION FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_obs.npz'),
        **{'training': train_result[1], 'testing': test_result[1]},
    )

    # LOSS FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_loss.npz'),
        **{'training': train_result[2], 'testing': test_result[2]},
    )

    # performance metrics write file
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_metrics.npz'),
        **{'training': train_result[3], 'testing': test_result[3]},
    )

    # MAE / ERROR FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_error.npz'),
        **{'training': train_result[4], 'testing': test_result[4]},
    )

    # LATENT MODEL FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_z.npz'),
        **{'training': train_result[5], 'testing': test_result[5]},
    )
    
    pickle_path = os.path.join(rpath, f'params_{model_instance}.pkl')
    with open(pickle_path, "wb") as file:
        pickle.dump(
            { 'training': nnx.state( train_result[6] ),
              'testing': nnx.state( test_result[6] )}, 
            file
        )
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_instance', type=int, default = 0)
    parser.add_argument('--latent_size', type=int, default = 30)
    parser.add_argument('--LB_factor', type=int, default = -5)

    args = parser.parse_args()
    model_instance = args.model_instance
    latent_size = args.latent_size
    LB_factor = args.LB_factor
    experiment = 'regselect'

    # set up rng state
    seed_key = random.PRNGKey(1234)
    runs = np.array((-8,-7,-6,-5,-4,-3,-2,-1))
    i = np.argwhere(runs==LB_factor).item()

    num_runs = len(runs)
    run_keys = random.split(seed_key,num_runs)
    this_run_key = run_keys[i]
    num_instances = 100
    model_keys = random.split(this_run_key, num_instances)
    model_key = model_keys[model_instance]
    
    main( experiment, Net, model_instance, latent_size, LB_factor, model_key )
