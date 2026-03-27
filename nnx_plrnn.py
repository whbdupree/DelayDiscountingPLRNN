from functools import partial
import pickle
import optax
from jax import tree,random, numpy as jnp
from flax import nnx

tree_map = partial( tree.map,is_leaf=lambda x: isinstance(x, nnx.Pytree))

class PLRNN(nnx.Module):
    def __init__(self, latent_size: int, rngs: nnx.Rngs, scale=10):
        AW_key = rngs()
        R = random.normal(AW_key, (latent_size,latent_size))
        num = ((R.T @ R) / latent_size) + (scale * jnp.identity(latent_size))
        w, v = jnp.linalg.eigh(num)
        denom = jnp.max(w.real)
        H = num / denom
        A = jnp.diag(H)
        W = jnp.fill_diagonal(H, 0, inplace=False)

        self.A = nnx.Param(A)
        self.Wh = nnx.Linear(
            latent_size,
            latent_size,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        # overwrite linear kernel with custom initialization
        self.Wh.kernel = nnx.Param(W)

    def __call__(self, z):
        Wh_terms = self.Wh(nnx.relu(z))
        A_term = z * self.A
        return A_term + Wh_terms

class LRNN(nnx.Module):
    def __init__(self, latent_size: int, rngs: nnx.Rngs, scale=10):
        AW_key = rngs()
        R = random.normal(AW_key, (latent_size,latent_size))
        num = ((R.T @ R) / latent_size) + (scale * jnp.identity(latent_size))
        w, v = jnp.linalg.eigh(num)
        denom = jnp.max(w.real)
        H = num / denom
        A = jnp.diag(H)
        W = jnp.fill_diagonal(H, 0, inplace=False)

        self.A = nnx.Param(A)
        self.Wh = nnx.Linear(
            latent_size,
            latent_size,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        # overwrite linear kernel with custom initialization
        self.Wh.kernel = nnx.Param(W)

    def __call__(self, z):
        Wh_terms = self.Wh(z)
        A_term = z * self.A
        return A_term + Wh_terms
    

class Net(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        session_sizes,
        latent_size,
        contexts,
        length,
    ):
        self.latent_model = PLRNN(latent_size, rngs=rngs)
        self.session_sizes = session_sizes
        self.latent_size = latent_size
        self.contexts = contexts
        self.length = length

        z0_init = nnx.initializers.normal()
        z0_key = rngs()
        z0 = z0_init(z0_key, (len(self.session_sizes), self.contexts, self.latent_size))
        self.z0 = nnx.Param(z0)

        linear = partial(nnx.Linear, latent_size, use_bias=False, rngs=rngs)
        self.B_tree = tree_map(linear, self.session_sizes)

    def __call__(self):
        length_minus = self.length - 1
        @nnx.scan(
            in_axes=nnx.Carry, out_axes=(nnx.Carry, -2), length=length_minus
        )  # scan time axis
        def forward(z):
            z = self.latent_model(z)
            return z, z

        z = jnp.zeros(
            ( len(self.session_sizes),
              self.contexts,
              self.length,
              self.latent_size, )
        )
        z = z.at[..., 0, :].set(self.z0)
        z_last, z_length_minus = forward( jnp.copy(self.z0) )
        z = z.at[..., 1:, :].set(z_length_minus)
        z_tree = tree.unflatten(tree.structure(self.session_sizes), z)

        linear_call = lambda a, b: a(b)
        x_tree = tree_map(linear_call, self.B_tree, z_tree)

        return x_tree, z

class linear_Net(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        session_sizes,
        latent_size,
        contexts,
        length,
    ):
        self.latent_model = LRNN(latent_size, rngs=rngs)
        self.session_sizes = session_sizes
        self.latent_size = latent_size
        self.contexts = contexts
        self.length = length

        z0_init = nnx.initializers.normal()
        z0_key = rngs()
        z0 = z0_init(z0_key, (len(self.session_sizes), self.contexts, self.latent_size))
        self.z0 = nnx.Param(z0)

        linear = partial(nnx.Linear, latent_size, use_bias=False, rngs=rngs)
        self.B_tree = tree_map(linear, self.session_sizes)

    def __call__(self):
        length_minus = self.length - 1
        @nnx.scan(
            in_axes=nnx.Carry, out_axes=(nnx.Carry, -2), length=length_minus
        )  # scan time axis
        def forward(z):
            z = self.latent_model(z)
            return z, z

        z = jnp.zeros(
            ( len(self.session_sizes),
              self.contexts,
              self.length,
              self.latent_size, )
        )
        z = z.at[..., 0, :].set(self.z0)
        z_last, z_length_minus = forward( jnp.copy(self.z0) )
        z = z.at[..., 1:, :].set(z_length_minus)
        z_tree = tree.unflatten(tree.structure(self.session_sizes), z)

        linear_call = lambda a, b: a(b)
        x_tree = tree_map(linear_call, self.B_tree, z_tree)

        return x_tree, z
    
    
class obsNet(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        session_sizes,
        latent_size,
    ):
        self.session_sizes = session_sizes
        self.latent_size = latent_size
        linear = partial(nnx.Linear, self.latent_size, use_bias=False, rngs=rngs)
        self.B_tree = tree_map(linear, self.session_sizes)
    def __call__(self,z):
        z_tree = tree.unflatten(tree.structure(self.session_sizes), z)
        linear_call = lambda a, b: a(b)
        x_tree = tree_map(linear_call, self.B_tree, z_tree)
        return x_tree

if __name__ == '__main__':
    seed = 123
    key = random.PRNGKey( seed )
    rngs = nnx.Rngs(key)
    T = 151
    contexts = 2
    latent_size = 5
    epochs = 3
    o1 = random.uniform( rngs(), (contexts,T,5) )
    o2 = random.uniform( rngs(), (contexts,T,7) )    
    obs = {'o1':o1,'o2':o2}
    session_sizes = tree.map( lambda a: a.shape[-1], obs)
    model = Net(rngs, session_sizes, latent_size, contexts, T)
    params = nnx.All(nnx.Param)
    diff_state = nnx.DiffState(0, params)
    optimizer = nnx.ModelAndOptimizer(
        model,
        tx=optax.chain(optax.clip(0.2), optax.adam(learning_rate=1e-3)),
        wrt=params,
    )

    def compute_loss(model,obs):
        x_tree, _ = model()
        error = tree.map(lambda a, b: jnp.abs(a - b), obs, x_tree)
        MAE = jnp.mean(jnp.concatenate(tree.leaves(error), axis=-1))
        return MAE, MAE
    loss_grad = nnx.value_and_grad( compute_loss, argnums=diff_state, has_aux=True)

    @nnx.scan(in_axes=nnx.Carry, out_axes=(0, 0, nnx.Carry), length=epochs)
    def optimizer_scan(optimizer):
        (loss, MAE), grads = loss_grad(optimizer.model, obs)
        grads.latent_model.Wh.kernel.value = jnp.fill_diagonal(
            grads.latent_model.Wh.kernel.value,
            0,
            inplace=False
        )
        optimizer.update(grads)
        return loss, MAE, optimizer

    MAE,_,optimizer = optimizer_scan(optimizer)
