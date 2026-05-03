#
# Effective label shift and correction: MNIST-NTK Example
#
# Execution code
#
import os
#os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import sys
from math import *
import numpy as np
import jax.numpy as jnp

import neural_tangents as nt
import matplotlib.pyplot as plt

from model import SmallConvNet, off_predict, on_predict, calc_eff_label, calc_label_correction, calc_iter_label_correction

from data import load_mnist_data

from plot import plot_errors_perfs, plot_label_shift_35, plot_label_shift_0to4


def get_frozen_seed(ik):
    return np.random.default_rng( (ik+1)*42 )


def calc_error_accuracy(Ypred, Ytarget):
    error = np.mean( np.multiply(Ypred - Ytarget, Ypred - Ytarget) )
    accuracy = np.mean( np.argmax(Ypred, axis=0) == np.argmax(Ytarget, axis=0)  ) 
    
    return error, accuracy


def calc_kernels(kernel_fn, Xtrain, Xtest, hy_params):
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']

    Kdata = np.zeros((Ntrain, Ntrain))
    Kpred = np.zeros((Ntrain, Ntest))

    for i in range(Ntrain//Nb):
        for j in range(Ntrain//Nb):
            Kdata[i*Nb:(i+1)*Nb, j*Nb:(j+1)*Nb] = kernel_fn(Xtrain[i*Nb:(i+1)*Nb], Xtrain[j*Nb:(j+1)*Nb], 'ntk')
        for j in range(Ntest//Nb):
            Kpred[i*Nb:(i+1)*Nb, j*Nb:(j+1)*Nb] = kernel_fn(Xtrain[i*Nb:(i+1)*Nb], Xtest[j*Nb:(j+1)*Nb], 'ntk')
        print('kernel_estimate : ', i)

    return Kdata, Kpred

# Cache the (Kdata, Kpred, indices) bundle for replicate `ik` under kernels/.
# First call samples fresh indices via `rng` and runs calc_kernels; later calls
# reload both the kernel and the indices, so the X/Y subsamples returned match
# the realization that was cached. 
def load_or_calc_kernels(kernel_fn, train_images, train_labels, test_images, test_labels,
                         hy_params, ik, rng):
    Ntrain, Ntest = hy_params['Ntrain'], hy_params['Ntest']
    task_type = hy_params.get('task_type', 'vanilla')
    cache_path = f'kernels/conv_ntk_kernel_{task_type}_Ntrain{Ntrain}_Ntest{Ntest}_ik{ik}.npz'

    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        Kdata = cache['Kdata']
        Kpred = cache['Kpred']
        train_idxs = cache['train_idxs']
        test_idxs = cache['test_idxs']
    else:
        train_idxs = rng.choice( range( len(train_images) ), Ntrain, replace=False )
        test_idxs = rng.choice( range( len(test_images) ), Ntest, replace=False )
        
        if task_type == 'class_inc' or 'task_inc':
            # Assumes the caller passed train_images already sorted by class or task (via make_class_inc_data)
            train_idxs = np.sort(train_idxs)            
        
        Xtrain = train_images[train_idxs, :, :, :]
        Xtest = test_images[test_idxs, :, :, :]
        Kdata, Kpred = calc_kernels(kernel_fn, Xtrain, Xtest, hy_params)
        os.makedirs('kernels', exist_ok=True)
        np.savez(cache_path, Kdata=Kdata, Kpred=Kpred,
                 train_idxs=train_idxs, test_idxs=test_idxs)

    Xtrain = train_images[train_idxs, :, :, :]
    Ytrain = train_labels[train_idxs, :]
    Xtest = test_images[test_idxs, :, :, :]
    Ytest = test_labels[test_idxs, :]
    return Kdata, Kpred, Xtrain, Ytrain, Xtest, Ytest


# Reorder full MNIST so all class-0 samples come first, then class-1, ..., class-9.
def make_class_inc_data(train_images, train_labels):
    int_labels = np.argmax(train_labels, axis=1)
    order = np.argsort(int_labels, kind='stable')
    return train_images[order], train_labels[order]
    

# Make the full MNIST into five binary classification tasks
def make_task_inc_data(rng, train_images, train_labels, test_images, test_labels):
    train_int = np.argmax(train_labels, axis=1)
    test_int = np.argmax(test_labels, axis=1)
    
    # Randomly shuffle the 10 classes
    classes = np.arange(10)
    rng.shuffle(classes)
    
    tasks = []
    # Create 5 binary tasks (pairs of classes)
    for i in range(0, 10, 2):
        c1, c2 = classes[i], classes[i+1]
        
        # --- Process Training Data ---
        train_mask = (train_int == c1) | (train_int == c2)
        x_train = train_images[train_mask]
        # Map: c1 -> 0, c2 -> 1, then to 2D one-hot
        y_train_binary = (train_int[train_mask] == c2).astype(np.int32)
        y_train = np.eye(2)[y_train_binary]
        
        # --- Process Test Data ---
        test_mask = (test_int == c1) | (test_int == c2)
        x_test = test_images[test_mask]
        # Use the SAME mapping: c1 -> 0, c2 -> 1, then to 2D one-hot
        y_test_binary = (test_int[test_mask] == c2).astype(np.int32)
        y_test = np.eye(2)[y_test_binary]
        
        tasks.append({
            'train_images': x_train,
            'train_labels': y_train,
            'test_images': x_test,
            'test_labels': y_test,
            'original_classes': (c1, c2)
        })
        
    t_train_images = np.vstack( [ task['train_images'] for task in tasks ] )
    t_train_labels = np.vstack( [ task['train_labels'] for task in tasks ] )
    t_test_images = np.vstack( [ task['test_images'] for task in tasks ] )
    t_test_labels = np.vstack( [ task['test_labels'] for task in tasks ] )
    
    return t_train_images, t_train_labels, t_test_images, t_test_labels, tasks 



# 
def run_exp_conv_ntk(hy_params):
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']
    
    init_fn, apply_fn, kernel_fn = SmallConvNet()
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    if hy_params['task_type'] == 'class_inc':
        train_images, train_labels = make_class_inc_data(train_images, train_labels)
    
    
    for ik in range( hy_params['ikmax'] ):
        rng = get_frozen_seed(ik)
        if hy_params['task_type'] == 'task_inc':
            train_images, train_labels, test_images, test_labels, tasks = make_task_inc_data(rng, train_images, train_labels, test_images, test_labels)
        
        Kdata, Kpred, Xtrain, Ytrain, Xtest, Ytest = load_or_calc_kernels(
            kernel_fn, train_images, train_labels, test_images, test_labels,
            hy_params, ik, rng)

        params_str = 'ttype_' + str(hy_params['task_type']) + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_ik' + str(ik)
        fgm_str = 'data/conv_ntk_gm_dep_' + params_str + '.csv'
        fgmw = open(fgm_str, 'w')

        gm_count = 20
        gms = np.logspace(-3, 3, gm_count)
        for gmidx, gm in enumerate(gms):
            hy_params['gm'] = gm
            yoff = off_predict(Kdata, Kpred, Ytrain, hy_params['gm'])
            error, perf = calc_error_accuracy(yoff, Ytest.T)
            fgmw.write( f"{gm:.6f}" + ',' + f"{error:.6f}" + ',' + f"{perf:.6f}" + '\n' )

        feta_str = 'data/conv_ntk_eta_dep_' + params_str + '.csv'
        fetaw = open(feta_str, 'w')

        eta_count = 20
        etas = np.logspace(-3, 1, eta_count) #np.logspace(-2, 1, eta_count)
        for etidx, eta in enumerate(etas):
            hy_params['eta'] = eta
            yon = on_predict(Kdata, Kpred, Ytrain, hy_params['eta'])
            error, perf = calc_error_accuracy(yon, Ytest.T)
            fetaw.write( f"{eta:.6f}" + ',' + f"{error:.6f}" + ',' + f"{perf:.6f}" + '\n' )
    

def visualize_label_shift(hy_params, with_correction):
    rng = get_frozen_seed( hy_params['ik'] )
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']

    init_fn, apply_fn, kernel_fn = SmallConvNet()
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    Kdata, Kpred, Xtrain, Ytrain, Xtest, Ytest = load_or_calc_kernels(
        kernel_fn, train_images, train_labels, test_images, test_labels,
        hy_params, hy_params['ik'], rng)

    Yeff = calc_eff_label(Kdata, Kpred, Ytrain, hy_params)
    
    if with_correction:
        Yc = calc_label_correction(Kdata, Kpred, Ytrain, hy_params)
    else:
        Yc = []
    
    Ytrue = Ytrain.T
    
    #plot_label_shift_35( rng, Ytrue, Yeff, Yc, hy_params, with_correction, Nplot=10 )
    plot_label_shift_0to4( rng, Ytrue, Yeff, Yc, hy_params, with_correction, Nplot=10 )
    
    

def run_learning_curve(hy_params):
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']
    gm, eta, ikmax = hy_params['gm'], hy_params['eta'], hy_params['ikmax']

    init_fn, apply_fn, kernel_fn = SmallConvNet()
    train_images, train_labels, test_images, test_labels = load_mnist_data()

    if hy_params['task_type'] == 'class_inc':
        train_images, train_labels = make_class_inc_data(train_images, train_labels)
    
    dN = 16
    dNiter = hy_params['dNiter']
    Ns = range(dN, Ntrain+1, dN)
    errors = np.zeros(( len(Ns), ikmax ))
    for ik in range( ikmax ):
        rng = get_frozen_seed(ik)
        if hy_params['task_type'] == 'task_inc':
            train_images, train_labels, test_images, test_labels, tasks = make_task_inc_data(rng, train_images, train_labels, test_images, test_labels)

        Kdata, Kpred, Xtrain, Ytrain, Xtest, Ytest = load_or_calc_kernels(
            kernel_fn, train_images, train_labels, test_images, test_labels,
            hy_params, ik, rng)

        Yc = ( calc_label_correction(Kdata, Kpred, Ytrain, hy_params) ).T
        
        Niters = range(dNiter, Ntrain+1, dNiter)
        Yc_iter = calc_iter_label_correction(Kdata, Kpred, Ytrain, Niters, hy_params)
        
        params_str = 'ttype_' + hy_params['task_type'] + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + '_dNiter' + str(dNiter) + '_ik' + str(ik)
        flc_str = 'data/conv_ntk_learn_curve_' + params_str + '.csv'
        fcw = open(flc_str, 'w')
        
        for Nidx, N in enumerate(Ns):
            yoff = off_predict(Kdata[:N, :N], Kpred[:N, :], Ytrain[:N, :], hy_params['gm'])
            off_error, off_perf = calc_error_accuracy(yoff, Ytest.T)
            
            yon = on_predict(Kdata[:N, :N], Kpred[:N, :], Ytrain[:N, :], hy_params['eta'])
            on_error, on_perf = calc_error_accuracy(yon, Ytest.T)
            
            yc = on_predict(Kdata[:N, :N], Kpred[:N, :], Yc[:N, :], hy_params['eta'])
            c_error, c_perf = calc_error_accuracy(yc, Ytest.T)
            
            yc_iter = on_predict(Kdata[:N, :N], Kpred[:N, :], Yc_iter[:N, :], hy_params['eta'])
            citer_error, citer_perf = calc_error_accuracy(yc_iter, Ytest.T)
            
            fcw.write( str(N) + ',' + f"{off_error:.6f}" + ',' + f"{off_perf:.6f}" + ',' + f"{on_error:.6f}"\
                        + ',' + f"{on_perf:.6f}" + ',' + f"{c_error:.6f}" + ',' + f"{c_perf:.6f}"\
                        + ',' + f"{citer_error:.6f}" + ',' + f"{citer_perf:.6f}" + '\n' )

            # training error
            if ik == 0:
                yon_train = on_predict(Kdata[:N, :N], Kdata[:N, :N], Ytrain[:N, :], hy_params['eta'])
                on_error_train, on_perf_train = calc_error_accuracy(yon_train, Ytrain[:N, :].T)
                yc_iter_train = on_predict(Kdata[:N, :N], Kdata[:N, :N], Yc_iter[:N, :], hy_params['eta'])
                citer_error_train, citer_perf_train = calc_error_accuracy(yc_iter_train, Ytrain[:N, :].T)
                print(N, on_error_train, citer_error_train)


if __name__ == "__main__":
    stdins = sys.argv # standard inputs
    
    #ik = int(stdins[1]) # the number of simulations

    #hyper parameters
    hy_params = {
        'Ntrain': 1024, #1024, # [512, 1024]
        'Ntest': 256, # 256
        'Nb': 16, # the block size for memory efficiency
        
        'gm': 0.01, 
        'eta': 1.0,
        'gm_z': 0.0, # for regularizer of the iterative correction
        'dNiter': 16, # batch size of iterative label correction
        
        'task_type': 'vanilla', # 'vanilla' or 'class_inc' or 'task_inc'
        
        'ikmax': 10, # maximum number of seeds
    }
    
    # parameter dependence (eta and gm)
    for ttype in ['vanilla', 'class_inc']: #['task_inc']: 
        hy_params['task_type'] = ttype
        run_exp_conv_ntk(hy_params)
    
    # visualization of effective label shift and correction 
    hy_params['ik'] = 0
    visualize_label_shift(hy_params, with_correction=False)
    visualize_label_shift(hy_params, with_correction=True)
    
    # learning curve estimation
    run_learning_curve(hy_params)
    
    # dNiter (batch size for iterative label correction) dependence
    dNiters = [4,16,64,256]
    for dNiter in dNiters:
        hy_params['dNiter'] = dNiter
        run_learning_curve(hy_params)
    
    # gm (regularizer amplitude) dependence
    gms = [0.001, 0.01, 0.1, 1.0, 10.0]
    for gm in gms:
        hy_params['gm'] = gm
        run_learning_curve(hy_params)

    # class incremental setting
    #hy_params['task_type'] = 'class_inc'
    #hy_params['dNiter'] = 16    
    #hy_params['eta'] = 0.001
    #hy_params['gm'] = 0.001
    #run_learning_curve(hy_params)
    
