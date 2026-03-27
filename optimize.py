import optax
import os
import yaml
import argparse
import pickle
import numpy as np
from jax import numpy as jnp
from jax import tree, vmap, random
from flax import nnx
from scipy import integrate, stats, ndimage, io
from functools import partial
from utils import data_loader
from operator import itemgetter
from nnx_plrnn import Net,linear_Net

tree_map = partial( tree.map,is_leaf=lambda x: isinstance(x, nnx.Pytree))

def compute_r2(obs,x):
    # assume obs,x are 1-dimensional
    r = obs-x
    SSres = jnp.sum( r*r )
    t = obs-jnp.mean(obs)
    SStot = jnp.sum( t*t )
    return 1-(SSres/SStot)
                    
def scramble(rngs,mm):
    off_diag =  ~jnp.eye( mm.shape[0],dtype=bool )
    rr = random.permutation( rngs(), mm[off_diag] )
    mm = mm.at[off_diag].set( rr )
    return mm
                    
def matrix_regularization( M, l1_ratio  ):
    flatM = M.reshape(-1)
    reg1 = jnp.linalg.norm(flatM, ord=1)
    reg2 = jnp.linalg.norm(flatM) ** 2
    reg =  l1_ratio*reg1 + (1-l1_ratio)*reg2
    return reg

def compute_loss(model, obs, L_Breg, l1_ratio):
    x_tree, _ = model()
    error = tree.map(lambda a, b: jnp.abs(a - b), obs, x_tree)
    MAE = jnp.mean(jnp.concatenate(tree.leaves(error), axis=-1))
    ls_B = tree_map(lambda x: x.kernel.value, model.B_tree)
    catB = jnp.concatenate(tree.leaves(ls_B), axis=1)
    Breg = L_Breg * matrix_regularization( catB, l1_ratio )
    loss = MAE + Breg
    return loss, MAE

def run_model(optimizer, loss_partial, epochs, obs, diff_state, train_bool = None):
    loss_grad = nnx.value_and_grad(loss_partial, argnums=diff_state, has_aux=True)
    if train_bool:
        @nnx.scan(in_axes=nnx.Carry, out_axes=(0, 0, nnx.Carry), length=epochs)
        def optimizer_scan(optimizer):
            (loss, MAE), grads = loss_grad(optimizer.model, obs)
            grads.latent_model.Wh.kernel.value = jnp.fill_diagonal(
                grads.latent_model.Wh.kernel.value,
                0, inplace=False,
            )
            optimizer.update(grads)
            return loss, MAE, optimizer
    else:
        @nnx.scan(in_axes=nnx.Carry, out_axes=(0, 0, nnx.Carry), length=epochs)
        def optimizer_scan(optimizer):
            (loss, MAE), grads = loss_grad(optimizer.model, obs)
            optimizer.update(grads)
            return loss, MAE, optimizer
    losses, MAEs, optimizer = optimizer_scan(optimizer)
    x, z = optimizer.model()
    return x, z, losses, MAEs, optimizer.model


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

    # DEFINE OPTIMIZERS

    train_params = nnx.All(nnx.Param)
    diff_state_train = nnx.DiffState(0, train_params)


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
    x_train, z_train, loss_train, MAEs_train, model_train = run_model(
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
    x_test, z_test, loss_test, MAEs_test, model_test = run_model(
        optimizer_test, loss_partial, test_epochs, obs_test, diff_state_test, train_bool=False
    )

    ccs_test = tree.map(get_ccs, obs_test, x_test)
    r2s_test = tree.map(get_r2s, obs_test, x_test)
    
    # #######Scramble W Control#########
    
    # overwrite
    model_SW.latent_model.Wh.bias = model_train.latent_model.Wh.bias
    model_SW.latent_model.A = model_train.latent_model.A
    W = jnp.copy( model_train.latent_model.Wh.kernel.value )
    SW = scramble( rngs, W  )
    # ^^ Scrambled W
    model_SW.latent_model.Wh.kernel.value = SW
    
    optimizer_SW = nnx.ModelAndOptimizer(
        model_SW,
        tx=optax.chain(optax.clip(0.2), optax.adam(learning_rate=1e-3)),
        wrt=test_params,
    )
    
    # run
    x_SW, z_SW, loss_SW, MAEs_SW, model_SW = run_model(
        optimizer_SW, loss_partial, test_epochs, obs_test, diff_state_test, train_bool=False
    )

    ccs_SW = tree.map(get_ccs, obs_test, x_SW)
    r2s_SW = tree.map(get_r2s, obs_test, x_SW)
    
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
    SW_result = (
        x_SW,
        obs_test,
        loss_SW,
        (ccs_SW,r2s_SW),
        MAEs_SW,
        z_SW,
        model_SW,
    )

    return train_result, test_result, SW_result, obs_keys_train, obs_keys_test


def main(experiment_name,Net,model_instance,latent_size,LB_factor,key):
    rngs = nnx.Rngs(key)
    
    l1_ratio = 0.5
    L_Breg = 10**LB_factor

    # output filenampe

    
    epath = os.path.join(os.getcwd(), 'results', experiment_name)
    
    os.makedirs(epath, exist_ok=True)
    rpath_relative = ('results',experiment_name,f'run{int(latent_size)}')
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
    
    train_result, test_result, SW_result, train_idxs, test_idxs = ml_passes(
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
        **{'training': train_result[0], 'testing': test_result[0], 'scramble':SW_result[0]},
    )

    # OBSERVATION FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_obs.npz'),
        **{'training': train_result[1], 'testing': test_result[1],'scramble':SW_result[1]},
    )

    # LOSS FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_loss.npz'),
        **{'training': train_result[2], 'testing': test_result[2],'scramble':SW_result[2]},
    )

    # performance metrics write file
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_metrics.npz'),
        **{'training': train_result[3], 'testing': test_result[3],'scramble':SW_result[3]},
    )

    # MAE / ERROR FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_error.npz'),
        **{'training': train_result[4], 'testing': test_result[4],'scramble':SW_result[4]},
    )

    # LATENT MODEL FILE WRITE
    np.savez(
        os.path.join(rpath, f'instance_{model_instance}_z.npz'),
        **{'training': train_result[5], 'testing': test_result[5],'scramble':SW_result[5]},
    )
    
    pickle_path = os.path.join(rpath, f'params_{model_instance}.pkl')
    with open(pickle_path, "wb") as file:
        pickle.dump(
            { 'training': nnx.state( train_result[6] ),
              'testing': nnx.state( test_result[6] ),
              'scramble': nnx.state( SW_result[6] )},
            file
        )
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_instance', type=int, default = 0)
    parser.add_argument('--latent_size', type=int, default = 30)
    parser.add_argument('--LB_factor', type=int, default = -5)
    parser.add_argument('--experiment',type=str,default='main')
    args = parser.parse_args()
    model_instance = args.model_instance
    latent_size = args.latent_size
    LB_factor = args.LB_factor
    experiment = args.experiment
    net_dict = { 'main':Net,
                 'linear':linear_Net }
    Net = net_dict[experiment]
    
    # set up rng state
    seed_key = random.PRNGKey(1234)
    runs = np.array((3,5,10,20,30,40,50,100,200,500,1000))
    i = np.argwhere(runs==latent_size).item()
    num_runs = len(runs)
    run_keys = random.split(seed_key,num_runs)
    this_run_key = run_keys[i]
    num_instances = 100
    model_keys = random.split(this_run_key, num_instances)
    model_key = model_keys[model_instance]
    
    main( experiment, Net, model_instance, latent_size, LB_factor, model_key )
