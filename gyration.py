import argparse
import os
import numpy as np
from jax import tree
from matplotlib import pyplot as plt
from matplotlib import patches
from utils import data_loader

plt.rcParams.update({
    'font.family':'sans-serif',
})

def process(u):
    '''
    This function is adapted from code provided at https://github.com/NevVerVer/neural-dynamics-gyration, and it is licensed as follows:

    MIT License

    Copyright (c) 2023 Ekaterina Kuzmina
    Copyright (c) 2026 William Barnett

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    '''
    us = u.shape
    if len(us) == 4:
        sessions,contexts,T,_ = us
        u = u.reshape( sessions*contexts,T,latent_size )
    elif len(us) == 3:
        contexts,T,_ = us
    else:
        print( 'Wrong input shape!!' )
        
    def soft_normalize(datas, thtrsh = 1e-2):
        fr_range = np.max(datas, axis=(0,1)) - np.min(datas, axis=(0,1))
        fr_range = np.where(fr_range > thtrsh, fr_range, 1.)
        datas /= fr_range
        return datas
    uu = np.mean( u, axis=0 )
    u = soft_normalize( u-uu )
    udot = np.diff( u, axis=1 )
    
    # compute gyration number
    X = np.concatenate( u[:,:-1] )        
    Xdot = np.concatenate( udot )
    Xeig = np.linalg.eigvals( Xdot.T @ X, )
    xnom = np.abs(np.real(Xeig[0])) + np.abs(np.real(Xeig[1]))
    ynom = np.abs(np.imag(Xeig[0])) + np.abs(np.imag(Xeig[1]))
    x = xnom / (np.abs(Xeig).sum())
    y = ynom / (np.abs(Xeig).sum())
    return x,y

def main(experiment, latent_size):
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{latent_size}' )

    num_instances = 100
    decay = np.zeros(num_instances)
    rotation = np.zeros(num_instances)
    for n in range(num_instances):
        zpath = os.path.join( rpath, f'instance_{n}_z.npz' )
        npz = np.load(zpath)
        z = npz['testing']
        x,y = process( z )
        decay[n] = x
        rotation[n] = y
    #
    obs_tree = data_loader()
    obs = np.concatenate( tree.leaves( obs_tree ),axis=-1 )
    obs_decay,obs_rotation = process( obs )
    
    fig = plt.figure()
    alpha = 0.5
    radius = 1    
    ax1 = plt.subplot(221)
    ax1.set_aspect('equal')
    ax1.grid(visible=True,which='major')
    #
    ax1.plot( (0,1), (0,1), '--k',
             linewidth = 0.5,
             alpha=alpha )
    arc = patches.Arc((0,0),radius*2,radius*2, # unit circle
                      theta1=0, theta2=90,
                      linestyle='--',
                      linewidth = 0.5,
                      color='k', alpha=alpha )
    ax1.add_patch(arc)
    #
    plot1, = ax1.plot(decay,rotation,'.',color='C2')
    plot2, = ax1.plot(obs_decay,obs_rotation,'.',
                  markeredgecolor='C3',
                  markerfacecolor='none')
    ax1.set_ylabel('Rotation Axis')
    ax1.set_xlabel('Decay Axis')
    ax1.set_xticks((0,0.5,1))
    ax1.set_yticks((0,0.5,1))    
    ax1.set_title('Gyration Plane')
    
    ax1.legend(
        (plot1,plot2),
        ('Latent Models','Observation'),
        handlelength = 0.5,
        handletextpad = 0.5,
        loc = 'lower right'
    )
    ax2 = plt.subplot(265)
    percent_change = 100*(rotation-obs_rotation)/obs_rotation
    print( 'max percent change:', np.max(percent_change) )
    print( 'min percent change:', np.min(percent_change) )    
    ax2.boxplot(
        percent_change,
        #showfliers = False,
        notch=True,
        bootstrap=1000,
    )
    ax2.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.set_ylabel('Percent greather than\nObservation')
    ax2.set_xticks((1,),('',))
    fig_path = os.path.join( fpath, f'gyration_D{latent_size}.pdf' )
    plt.savefig( fig_path )
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int, default = 30)    
    args = parser.parse_args()
    latent_size = args.dim_z    

    experiment = 'main'
    main(experiment,latent_size)
