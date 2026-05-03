#
# Effective label shift and correction: MNIST-NTK Example
#
# Model description
#
import sys
import numpy as np

import jax
import jax.numpy as jnp

from neural_tangents import stax 


# prediction of the offline kernel regression given data (X,y)
def off_predict(Kdata, Kpred, Ytrain, gm):

    In = jnp.eye( len(Ytrain) )
    alpha = jnp.linalg.solve( gm*In + Kdata, Kpred )
    
    return jnp.dot( Ytrain.T, alpha )
    

# prediction of the online model given data (X, y)
def on_predict(Kdata, Kpred, Ytrain, eta):

    KdataU = jnp.triu(Kdata, k=1)
    In = jnp.eye( len(Ytrain) )

    alpha = jnp.linalg.solve( (1/eta)*In + KdataU, Kpred )
    
    return np.dot( Ytrain.T, alpha )


################################
# Label shift and correction   #
################################

# effective label of online kernel regression
def calc_eff_label(Kdata, Kpred, Ytrain, hy_params):
    eta, gm = hy_params['eta'], hy_params['gm']
    
    KdataU = jnp.triu(Kdata, k=1)
    In = jnp.eye( len(Ytrain) )
    Mon = (1/eta) * In + KdataU # (N, N)
    Moff = gm * In + Kdata
    
    C_label_shift = jnp.linalg.solve(Mon, Moff)
    
    return jnp.dot( Ytrain.T, C_label_shift)


def calc_label_correction(Kdata, Kpred, Ytrain, hy_params):
    eta, gm = hy_params['eta'], hy_params['gm']
    
    KdataU = jnp.triu(Kdata, k=1)
    In = jnp.eye( len(Ytrain) )
    Mon = (1/eta) * In + KdataU # (N, N)
    Moff = gm * In + Kdata
    
    C_label_correction = jnp.linalg.solve(Moff, Mon)
    
    return jnp.dot( Ytrain.T, C_label_correction )
    

def calc_iter_label_correction(Kdata, Kpred, Ytrain, Niters, hy_params):
    eta, gm, gm_z = hy_params['eta'], hy_params['gm'], hy_params['gm_z']
    
    KdataU = jnp.triu( Kdata, k=1 )
    Yoff = ( off_predict( Kdata, Kdata, Ytrain, gm ) ).T
    
    Yc_iter = np.zeros(( np.shape(Ytrain) ))
    dN = Niters[1] - Niters[0]
    
    for Nidx, N in enumerate(Niters):
        if Nidx == 0:
            Yc_iter[:N,:] = Ytrain[:N,:]
        else:
            Ip = jnp.eye(N-dN)
            In = jnp.eye(dN)
    
            Kdata_pp  = Kdata[:N-dN, :N-dN]
            Kdata_pn  = Kdata[:N-dN, N-dN:N]
            Kdata_nn  = Kdata[N-dN:N, N-dN:N]
            
            KdataU_nn = KdataU[N-dN:N, N-dN:N]
    
            Yon_new = ( on_predict( Kdata_pp, Kdata_pn, Yc_iter[:N-dN, :], eta ) ).T
            dYon_new = Ytrain[N-dN:N,:] - Yon_new
            Yoff_new = ( off_predict( Kdata_pp, Kdata_pn, Ytrain[:N-dN,:], gm ) ).T
            dYoff_new = Yoff_new - Ytrain[N-dN:N,:]
            
            Citer = jnp.linalg.solve( gm_z*In + Kdata_nn, (1/eta)*In + KdataU_nn ) 
            Con = Citer - In
            
            Q = jnp.dot( Kdata_pn.T, jnp.linalg.solve(gm*Ip + Kdata_pp, Kdata_pn)  )
            Coff = gm * jnp.linalg.solve( (gm*In + Kdata_nn) - Q, Citer )
            
            Yc_iter[N-dN:N,:] = Ytrain[N-dN:N,:] + jnp.dot(dYon_new.T, Con).T + jnp.dot(dYoff_new.T, Coff).T
            
    return Yc_iter
    


################################
# Functions for NTK estimation #
################################

def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
        stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
        stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))

    Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
        channels, (3, 3), strides, padding='SAME')

    return stax.serial(stax.FanOut(2),
                        stax.parallel(Main, Shortcut),
                        stax.FanInSum())

def WideResnetGroup(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [WideResnetBlock(channels, (1, 1))]
    return stax.serial(*blocks)


def WideResnet(block_size, k, num_classes):
    return stax.serial(
        stax.Conv(16, (3, 3), padding='SAME'),
        WideResnetGroup(block_size, int(16 * k)),
        WideResnetGroup(block_size, int(32 * k), (2, 2)),
        WideResnetGroup(block_size, int(64 * k), (2, 2)),
        stax.AvgPool((8, 8)),
        stax.Flatten(),
        stax.Dense(num_classes, 1., 0.))


def SmallConvNet(num_classes=10):
    return stax.serial(
        stax.Conv(32, (3,3), padding='SAME'),
        stax.Relu(), stax.AvgPool((2,2), strides=(2, 2)),
        stax.Conv(64, (3,3), padding='SAME'),
        stax.Relu(), stax.AvgPool((2,2), strides=(2, 2)),
        stax.Flatten(),
        stax.Dense(num_classes, 1., 0.))




