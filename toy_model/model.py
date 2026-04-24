#
# Simple illutrations of effective target shift and it correction
#
# Model description
#
import numpy as np


####################################
# online/offline Kernel regression #
####################################

# kernel estimation
def calc_kernel(X1, X2, params):
    if params['kernel'] == 'RBF':
        return calc_RBF(X1, X2, params['sig2'])
    elif params['kernel'] == 'RPK':
        return calc_RPK(X1, X2, params['W'])


# RBF kernel 
def calc_RBF(X1, X2, sig2):
    # input: X1: (B, Lx), X2: (B, Lx)
    # output: (B, B)
    dX = np.expand_dims(X1, axis=1) - np.expand_dims(X2, axis=0)
    dX2sum = np.sum( np.multiply(dX, dX), axis=2 )
    return np.exp( - (1.0/sig2) * dX2sum )


def fh(X, W):
    _, Lh = np.shape(W)
    return np.tanh( np.dot(X, W) )/np.sqrt(Lh)
    

# Random Projection kernel 
def calc_RPK(X1, X2, W):
    # input: X1: (B, Lx), X2: (B, Lx)
    # output: (B, B)
    # W : (Lx, Lh)
    H1 = fh(X1, W) # (B, Lh)
    H2 = fh(X2, W) # (B, Lh)
        
    return np.dot(H1, H2.T)


    
# prediction of the offline kernel regression given data (X,y)
def foff(X, y, Xnew, params):
    gm, sig2 = params['gm'], params['sig2']
    
    Kx_new = calc_kernel(X, Xnew, params)
    Kx_old = calc_kernel(X, X, params)
    
    In = np.eye( len(y) )
    C_old_new = np.linalg.solve( gm*In + Kx_old, Kx_new )
    
    return np.dot( y, C_old_new )
    

# prediction of the online model given data (X, y)
def fon(X, y, Xnew, params):
    eta = params['eta']
    
    Kx_new = calc_kernel(X, Xnew, params)
    Kx_old = calc_kernel(X, X, params)
    KxU = np.triu(Kx_old, k=1)
    
    In = np.eye( len(y) )
    C_old_new = np.linalg.solve( (1/eta)*In + KxU, Kx_new )
    
    return np.dot( y, C_old_new )
    

# prediction of the mini-batch online model given data (X, y)
def fmb(X, y, Xnew, params):
    eta, mb_size = params['eta'], params['mb_size']
    B, _ = np.shape(X)
    
    Kx_new = calc_kernel(X, Xnew, params)
    Kx_old = calc_kernel(X, X, params)
    KxUb = np.triu(Kx_old, k=1)
    
    # zero pad block diagonal parts
    # assert B%mb_size = 0; there's residual at boundary otherwise
    for i in range( B//mb_size ):
        KxUb[ i*mb_size:(i+1)*mb_size, i*mb_size:(i+1)*mb_size ] = np.zeros((mb_size, mb_size))
    
    In = np.eye( len(y) )
    C_old_new = np.linalg.solve( (1/eta)*In + KxUb, Kx_new )
    
    return np.dot( y, C_old_new )


# effective label of online kernel regression
def calc_eff_label(X, y, params):
    sig2, eta, gm = params['sig2'], params['eta'], params['gm']
    
    Kx = calc_kernel(X, X, params)
    KxU = np.triu(Kx, k=1)
    
    In = np.eye( len(y) )
    Mon = (1/eta) * In + KxU # (N, N)
    Moff = gm * In + Kx
    
    C_label_shift = np.linalg.solve(Mon, Moff)
    
    return np.dot(y, C_label_shift)


# X : (N, Lx)
# y : (N, Ly)
def calc_label_correction(X, y, params):
    eta, gm, sig2 = params['eta'], params['gm'], params['sig2']
    Kx_mat = calc_kernel(X, X, params)
    KxU = np.triu(Kx_mat, k=1)
    
    In = np.eye( len(y) )
    Mon = (1/eta) * In + KxU # (N, N)
    Moff = gm * In + Kx_mat
    C_label_correction = np.linalg.solve(Moff, Mon)
    
    return np.dot(y.T, C_label_correction)


#####################################################
# Online update                                     #
# implementation of online update for sanity check  #
#####################################################


def simul_fon(W, Wout, X):
    H = fh(X, W) # (B, Lh)
    return np.dot(H, Wout) # (B)


# online update
def on_update_Wout(W, Wout, X, Y, eta): 
    Yon = simul_fon(W, Wout, X)
    HX = fh(X, W)
    Wout = Wout - eta * (Yon - Y) * HX
    
    return Wout
    

# mini-batch update
def mb_update_Wout(W, Wout, X, Y, eta): 
    Yon = simul_fon(W, Wout, X)
    HX = fh(X, W)
    
    Wout = Wout - eta * np.dot(Yon - Y, HX)
    
    return Wout



##########################
# Performance evaluation #
##########################

def calc_error_accuracy(y1, y2):
    error = np.mean( np.multiply(y1-y2, y1-y2) )
    accuracy = np.mean( np.sign(y1) == np.sign(y2) )    
    return error, accuracy


def calc_error(y1, y2):
    return np.mean( np.multiply(y1-y2, y1-y2) )
    

# training and test error under offline learning
def calc_err_off(rng, Xtrain, Ytrain, Xtest, Ytest, params, data_type):
    Yoff_train = foff(Xtrain, Ytrain, Xtrain, params)
    Yoff_test = foff(Xtrain, Ytrain, Xtest, params)
    
    train_error = calc_error(Yoff_train, Ytrain)
    test_error = calc_error(Yoff_test, Ytest)
    
    return train_error, test_error


# training and test error under online learning
def calc_err_on(rng, Xtrain, Ytrain, Xtest, Ytest, params, data_type):
    Yon_train = fon(Xtrain, Ytrain, Xtrain, params)
    Yon_test = fon(Xtrain, Ytrain, Xtest, params)
    
    on_train_error = calc_error(Yon_train, Ytrain)
    on_test_error = calc_error(Yon_test, Ytest)
    
    return on_train_error, on_test_error 
