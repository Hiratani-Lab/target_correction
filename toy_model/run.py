#
# Simple illutrations of effective target shift and it correction
#
# Execution code
#
import numpy as np

from data import load_mcycle_data, generate_toyGP_data
from model import fon, foff, fmb, calc_eff_label, calc_label_correction, calc_err_off, calc_error, on_update_Wout, mb_update_Wout, simul_fon, calc_kernel
from plot import plot_sig2_gm_dependence, plot_label_shift_correction, plot_learning_curves


# estimate the optimal gamma and sig2 for offline learning
def calc_opt_sig2_gm(rng, params, data_type):
    Ntrain, num_seeds = params['Ntrain'], params['num_seeds']
    
    if data_type == 'toyGP':
        sig2_log_range = [-3, 0]
        gm_log_range = [-2, 2]
    if data_type == 'mcycle':
        sig2_log_range = [-1, 3]
        gm_log_range = [-3, 2]

    params['kernel'] = 'RBF'
    Mparams1 = 20
    Mparams2 = 30
    sig2s = np.logspace(sig2_log_range[0], sig2_log_range[1], Mparams1)
    gms = np.logspace(gm_log_range[0], gm_log_range[1], Mparams2)
    
    errors = { 'train': np.zeros((Mparams1, Mparams2, num_seeds)),\
                'test': np.zeros((Mparams1, Mparams2, num_seeds)) }
    
    for seed in range(num_seeds):
        params['seed'] = seed
        if data_type == 'toyGP':
            Xtrain, Ytrain, Xtest, Ytest = generate_toyGP_data(rng, params)
        elif data_type == 'mcycle':
            Xtrain, Ytrain, Xtest, Ytest = load_mcycle_data(rng, Ntrain)
        
        for sidx, sig2 in enumerate(sig2s):
            params['sig2'] = sig2
            for gidx, gm in enumerate(gms):
                params['gm'] = gm
                errors['train'][sidx, gidx, seed], errors['test'][sidx, gidx, seed] = calc_err_off(rng, Xtrain, Ytrain, Xtest, Ytest, params, data_type)
    
    plot_sig2_gm_dependence(params, errors, data_type, sig2s, gms)
    
    return errors
    

# vissualization of effective target shift and its correction

def visualize_label_shift_correction(rng, params, data_type):
    Ntrain = params['Ntrain']
    params['kernel'] = 'RBF'
    
    if data_type == 'toyGP':
        Xtrain, Ytrain, Xtest, Ytest = generate_toyGP_data(rng, params)
        Xs = np.expand_dims( np.arange(0, 1, 0.01), axis=1 )
    elif data_type == 'mcycle':
        Xtrain, Ytrain, Xtest, Ytest = load_mcycle_data(rng, Ntrain)
        Xs = np.expand_dims( np.arange(0, 60, 0.25), axis=1 )
        
    plot_label_shift_correction(rng, Xs, Xtrain, Ytrain, Xtest, Ytest, params, data_type)



def calc_learning_curve(rng, params, data_type):
    Ntrain, num_seeds = params['Ntrain'], params['num_seeds']
    
    params['kernel'] = 'RPK' # random projection
    params['Lh'] = 100
    params['W'] = rng.normal(0.0, 1.0, (1, params['Lh']))
    
    Nparts = range(1, Ntrain+1, 1)
                  
    if data_type == 'toyGP':
        Xtrain, Ytrain, Xtest, Ytest = generate_toyGP_data(rng, params)
    if data_type == 'mcycle':
        Xtrain, Ytrain, Xtest, Ytest = load_mcycle_data(rng, Ntrain)
    
    gms = np.array([0.03, 0.3, 3.0]) #np.array([0.03, 0.1, 0.3, 1.0, 3.0])
    
    if params['kernel'] == 'RBF':
        k_const = 1.0
    else:
        k_const = np.mean( np.diag( calc_kernel(Xtest, Xtest, params) ) )
        
    etas = 1/(k_const + gms)
    gmlen = len(gms)
    
    off_errs = {}
    for gmidx, gm in enumerate(gms):
        off_errs[gm] = {'train': np.zeros(len(Nparts)), 'test': np.zeros(len(Nparts))}
        params['gm'] = gm
        for Nidx, N in enumerate(Nparts):
            Xtrain_part = Xtrain[:N]
            Ytrain_part = Ytrain[:N]
            
            Yoff_train = foff(Xtrain_part, Ytrain_part, Xtrain_part, params)
            Yoff_test = foff(Xtrain_part, Ytrain_part, Xtest, params)
            
            off_errs[gm]['train'][Nidx] = calc_error(Yoff_train, Ytrain_part)
            off_errs[gm]['test'][Nidx] = calc_error(Yoff_test, Ytest)
    
    
    on_errs = {}
    on_simul_errs = {}
    for etidx, eta in enumerate(etas):
        on_errs[etidx] = {'train': np.zeros(len(Nparts)), 'test': np.zeros(len(Nparts))}
        on_simul_errs[etidx] = {'train': np.zeros(len(Nparts)), 'test': np.zeros(len(Nparts))}

        params['eta'] = eta
        Wout = np.zeros(( params['Lh'] )) # initialize readout weight
        
        for Nidx, N in enumerate(Nparts):
            Xtrain_part = Xtrain[:N]
            Ytrain_part = Ytrain[:N]
            
            Yon_train = fon(Xtrain_part, Ytrain_part, Xtrain_part, params)
            Yon_test = fon(Xtrain_part, Ytrain_part, Xtest, params)
            
            on_errs[etidx]['train'][Nidx] = calc_error(Yon_train, Ytrain_part)
            on_errs[etidx]['test'][Nidx] = calc_error(Yon_test, Ytest)
            
            Wout = on_update_Wout(params['W'], Wout, Xtrain[Nidx], Ytrain[Nidx], eta)
            Yon_simul_train = simul_fon(params['W'], Wout, Xtrain_part)
            Yon_simul_test = simul_fon(params['W'], Wout, Xtest)
            on_simul_errs[etidx]['train'][Nidx] = np.clip( calc_error(Yon_simul_train, Ytrain_part), 0.0, 1e3)
            on_simul_errs[etidx]['test'][Nidx] = np.clip( calc_error(Yon_simul_test, Ytest), 0.0, 1e3)
    
    mb_errs = {}
    mb_simul_errs = {}
    mb_sizes = [2,4,8]
    
    eta = 1.0
    params['eta'] = eta
    for mbidx, mb_size in enumerate(mb_sizes):
        mb_errs[mbidx] = {'mbNs' : np.arange(mb_size, Ntrain+1, mb_size), 'train': np.zeros(len(Nparts)//mb_size), 'test': np.zeros(len(Nparts)//mb_size)}
        mb_simul_errs[mbidx] = {'mbNs' : np.arange(mb_size, Ntrain+1, mb_size), 'train': np.zeros(len(Nparts)//mb_size), 'test': np.zeros(len(Nparts)//mb_size)}

        params['mb_size'] = mb_size
        Wout = np.zeros(( params['Lh'] )) # initialize readout weight
        
        for Nidx, N in enumerate( mb_errs[mbidx]['mbNs'] ):
            Xtrain_part = Xtrain[:N]
            Ytrain_part = Ytrain[:N]
            
            Ymb_train = fmb(Xtrain_part, Ytrain_part, Xtrain_part, params)
            Ymb_test = fmb(Xtrain_part, Ytrain_part, Xtest, params)
            
            mb_errs[mbidx]['train'][Nidx] = calc_error(Ymb_train, Ytrain_part)
            mb_errs[mbidx]['test'][Nidx] = calc_error(Ymb_test, Ytest)
            
            Xmb = Xtrain[Nidx*mb_size:(Nidx+1)*mb_size]
            Ymb = Ytrain[Nidx*mb_size:(Nidx+1)*mb_size]
            
            Wout = mb_update_Wout(params['W'], Wout, Xmb, Ymb, eta)
            Ymb_simul_train = simul_fon(params['W'], Wout, Xtrain_part)
            Ymb_simul_test = simul_fon(params['W'], Wout, Xtest)
            mb_simul_errs[mbidx]['train'][Nidx] = calc_error(Ymb_simul_train, Ytrain_part)
            mb_simul_errs[mbidx]['test'][Nidx] = calc_error(Ymb_simul_test, Ytest)
                        
    plot_learning_curves(Nparts, off_errs, on_errs, on_simul_errs, mb_errs, mb_simul_errs, gms, etas, mb_sizes, params)


if __name__ == "__main__":
    seed = 1 # pick random seeds for visualization
    # seed = 1 for calc_learning_curve
    # seed = 49 for visualize_label_shift_correction
    rng = np.random.default_rng(seed)
    
    params_shared = {
        'Ntrain': 40, # Number of samples used for training
        'num_seeds': 100,
        'm_seed': seed, # master seed
    }
    
    # parameters for toyGP curve dataset
    params_toyGP = {
        'noise_level': 0.3, #0.5, # the noise on the true targets
        'true_sig2': 0.1, # width of the true RBF kernel 
        
        'sig2': 0.1, # width of the RBF kernel
        'gm': 1.0, # # amplitude of L2 regularizer in the offline kernel regression
        'eta': 0.5,  #learning rate
    }
    
    # parameters for mcycle dataset
    params_mcycle = {
        'sig2': 50.0, #30.0, # width of the RBF kernel
        'gm': 1.0, # amplitude of L2 regularizer in the offline kernel regression
        'eta': 0.4, #0.3, #learning rate
    }
    
    params = params_shared | params_toyGP
    
    # dependence on kernel width (sig) and the regularization strength (gm)
    #calc_opt_sig2_gm(rng, params, 'toyGP')
    
    #visualize_label_shift_correction(rng, params, 'toyGP')
    
    calc_learning_curve(rng, params, 'toyGP')
    
