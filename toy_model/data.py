#
# Simple illutrations of effective target shift and it correction
#
# Data generation
#
import numpy as np
import pandas as pd


#######################################################################
# toy GP curve data
# sampling from a GP defined by a RBF kernel with width = true_sig
# targets are corrupted with Gaussian noise with sd = noise_level
#######################################################################

def generate_toyGP_curve(rng, Xs, sig2):
    dX = np.expand_dims(Xs, axis=1) - np.expand_dims(Xs, axis=0)
    dX2sum = np.multiply(dX, dX)
    SigX = np.exp( - (1.0/sig2) * dX2sum )
    
    Xzero = np.zeros( np.shape(Xs) )
    return rng.multivariate_normal( Xzero, SigX )


def generate_toyGP_data(rng, params):
    Ntrain, true_sig2, noise_level = params['Ntrain'], params['true_sig2'], params['noise_level']
    
    Xs = np.arange(0.0, 1.0, 0.005)
    Ys = generate_toyGP_curve(rng, Xs, true_sig2)
    
    Ntest = 200 # large test size
    idxtmp = rng.choice( range(len(Xs)), (Ntrain+Ntest) )
    train_idx = idxtmp[:Ntrain]
    test_idx = idxtmp[Ntrain:]
    
    Xtrain = Xs[train_idx]
    Ytrain = Ys[train_idx]
    
    Xtest = Xs[test_idx]
    Ytest = Ys[test_idx]
    
    Xtrain = np.expand_dims(Xtrain, axis=1) # (Ntrain) -> (Ntrain, 1)
    Xtest = np.expand_dims(Xtest, axis=1) # (Ntest) -> (Ntest, 1)
    
    # add noise to observations
    Ytrain = Ytrain + rng.normal( 0.0, noise_level, np.shape(Ytrain) )
    Ytest = Ytest + rng.normal( 0.0, noise_level, np.shape(Ytest) )
    
    return Xtrain, Ytrain, Xtest, Ytest


########################################################################
# Load mcycle data 
# random train/test splitting, no preprocessing
########################################################################
def load_mcycle_data(rng, Ntrain):
    file_name = 'dataset/dataset-72001.csv'
    mcycle_df = pd.read_csv(file_name)
    Xs = mcycle_df['times'].to_numpy()
    Ys = mcycle_df['accel'].to_numpy()
    
    Ntot = len(Xs)
    perm = rng.permutation(Ntot)

    Xs = Xs[perm]
    Ys = Ys[perm]
    
    Xtrain = Xs[:Ntrain]
    Xtest = Xs[Ntrain:]
    
    Ytrain = Ys[:Ntrain]
    Ytest = Ys[Ntrain:]
    
    Xtrain = np.expand_dims(Xtrain, axis=1) # (Ntrain) -> (Ntrain, 1)
    Xtest = np.expand_dims(Xtest, axis=1) # (Ntest) -> (Ntest, 1)
    
    return Xtrain, Ytrain, Xtest, Ytest
    


