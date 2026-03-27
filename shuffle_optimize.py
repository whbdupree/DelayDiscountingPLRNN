import optax
import os
import yaml
import argparse
import pickle
from jax import vmap,tree,random,numpy as jnp
from flax import nnx
from functools import partial
from optimize import compute_r2, matrix_regularization

from nnx_plrnn import obsNet,tree_map

def shuffle( rngs, T, interval_size ):
    tt=jnp.arange(T)
    C = int(T/interval_size)
    ttC = jnp.array_split(tt,C)
    Cr = random.permutation( rngs(), C )
    return jnp.concatenate( [ttC[c] for c in Cr] )

def compute_loss(model, z, obs, L_Breg, l1_ratio):
    x_tree = model(z)
    error = tree.map(lambda a, b: jnp.abs(a - b), obs, x_tree)
    MAE = jnp.mean(jnp.concatenate(tree.leaves(error), axis=-1))
    ls_B = tree_map(lambda x: x.kernel.value, model.B_tree)
    catB = jnp.concatenate(tree.leaves(ls_B), axis=1)
    Breg = L_Breg * matrix_regularization( catB, l1_ratio )
    loss = MAE + Breg
    return loss, MAE

def main( z, obs, shuffle_size, latent_size, LB_factor, rngs ):
    ## shuffle z in chunks of shuffle_size
    zshape = z.shape
    _,_,T,_ = zshape

    shuffle_idx = shuffle( rngs, T, shuffle_size )
    shuffle_z = z[...,shuffle_idx,:]
    
    ## build model
    session_sizes = tree.map( lambda a:a.shape[-1], obs)
    epochs = 3000
    model = obsNet(rngs,session_sizes,latent_size)
    
    ## set up optimizer
    params = nnx.All(nnx.Param)
    diff_state = nnx.DiffState(0, params)
    optimizer = nnx.ModelAndOptimizer(
        model,
        tx=optax.chain(optax.clip(0.2), optax.adam(learning_rate=1e-3)),
        wrt=params,
    )
    L_Breg = 10**LB_factor
    loss_partial = partial(
        compute_loss,
        L_Breg=L_Breg,
        l1_ratio=0.5,
    )

    loss_grad = nnx.value_and_grad(loss_partial, argnums=diff_state, has_aux=True)
    @nnx.scan(in_axes=(None,None,nnx.Carry), out_axes=(0, 0, nnx.Carry), length=epochs)
    def optimizer_scan(shuffle_z,obs,optimizer):
        (loss, MAE), grads = loss_grad(optimizer.model, shuffle_z, obs)
        optimizer.update(grads)
        return loss, MAE, optimizer
    
    ## go
    losses, MAEs, optimizer = optimizer_scan(shuffle_z,obs,optimizer)

    model = optimizer.model
    x_tree = model(shuffle_z)

    get_ccs = vmap(vmap(lambda a, b: jnp.corrcoef(a, b)[0, 1], in_axes=1))
    get_r2s = vmap(vmap(lambda a, b: compute_r2(a,b), in_axes=1))
    ccs = tree.map(get_ccs, obs, x_tree)
    r2s = tree.map(get_r2s, obs, x_tree)    
    out = { 'losses':losses,
            'errors':MAEs,
            'B_tree':nnx.state(model.B_tree),
            'ccs':ccs,
            'r2s':r2s, }
    return out
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_instance', type=int, default = 0)
    parser.add_argument('--latent_size', type=int, default = 30)    
    parser.add_argument('--LB_factor', type=int, default = -5)
    parser.add_argument('--experiment',type=str,default='main')
    args = parser.parse_args()
    model_instance = args.model_instance
    LB_factor = args.LB_factor
    latent_size = args.latent_size
    experiment_name = args.experiment
    
    num_instances = 100
    seed_key = random.PRNGKey(1111)
    
    runs = jnp.array((3,5,10,20,30,40,50,100,200,500,1000))
    i = jnp.argwhere(runs==latent_size).item() 

    num_runs = len(runs)
    
    run_keys = random.split(seed_key,num_runs)
    this_run_key = run_keys[i]

    model_keys = random.split(this_run_key, num_instances)
    model_key = model_keys[model_instance]

    rngs = nnx.Rngs(model_key)

    ## load things
    rpath_relative = ('results',experiment_name,f'run{int(latent_size)}')
    rpath = os.path.join(os.getcwd(),*rpath_relative)
    # load testing data
    opath = os.path.join( rpath, f'instance_{model_instance}_obs.npz')
    obsnpz = jnp.load( opath, allow_pickle=True )
    obs = obsnpz['testing'].item()
    # load latent factors from testing forward pass
    zpath = os.path.join( rpath, f'instance_{model_instance}_z.npz' )
    znpz = jnp.load( zpath )
    z = znpz['testing']
    shuffle_size = 5
    s5_results = main(
        z, obs, shuffle_size, latent_size, LB_factor, rngs )
    shuffle_size = 15
    s15_results = main(
        z, obs, shuffle_size, latent_size, LB_factor, rngs )
    write_path = os.path.join(
        rpath, f'shuffle_results_{model_instance}.pkl' )
    with open( write_path, 'wb') as file :
        pickle.dump(
            {'shuffle5':s5_results,'shuffle15':s15_results},
            file
        )
