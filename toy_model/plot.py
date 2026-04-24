#
# Simple illutrations of effective target shift and it correction
#
# Plotting
#
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("ggplot")
plt.rcParams.update({'font.size':16})

lss = {'train': '--', 'test': '-'} # training curves are dashed, test curves are solid
ltype_clrs = {'off' : 'tab:blue', 'on': 'tab:orange', 'plus': 'purple', 'c' : 'tab:green'}

from model import fon, foff, calc_eff_label, calc_label_correction

# plottint the optimal parameters (sig2 and gm)
def plot_sig2_gm_dependence(params, errors, data_type, sig2s, gms):
    Mparams1 = len(sig2s)
    Mparams2 = len(gms)
    
    sig2_grid = np.logspace( np.log10(sig2s[0]), np.log10(sig2s[-1]), Mparams1+1)
    gm_grid = np.logspace( np.log10(gms[0]), np.log10(gms[-1]), Mparams2+1)
    X, Y = np.meshgrid(gm_grid, sig2_grid)
    
    mean_errors = {}
    for key in errors.keys():
        mean_errors[key] = np.mean(errors[key], axis=2)
    
    fig = plt.figure(figsize=(10.8, 4.8))
    
    for kidx, key in enumerate( mean_errors.keys() ):
        plt.subplot(1,2,kidx+1)
        plt.title(key)
        plt.pcolor(X, Y, np.log(mean_errors[key]) )
        plt.loglog()
        plt.colorbar()
        plt.xlabel('gm')
        plt.ylabel('sig2')
        
    min_idxs = np.unravel_index(mean_errors['test'].argmin(), mean_errors['test'].shape)
    print( 'gm:', gms[ min_idxs[1] ], ', sig2:', sig2s[ min_idxs[0] ] )
    
    plt.show()
    
    params_str = '_Ntrain' + str(params['Ntrain']) + '_nsd' + str(params['num_seeds']) + '_nlvl' + f"{params['noise_level']:.2f}"\
                 + '_tsg2_' + f"{params['true_sig2']:.2f}"
    fig.savefig('figs/' + data_type + '_opt_sig2_gm' + params_str + '.pdf')
    


def plot_label_shift_correction(rng, Xs, Xtrain, Ytrain, Xtest, Ytest, params, data_type):
    sig2, gm, eta = params['sig2'], params['gm'], params['eta']
    
    # effective and corrected Ytrain
    eff_Ytrain = calc_eff_label(Xtrain, Ytrain, params)
    Yc_train = calc_label_correction(Xtrain, Ytrain, params)
    
    # pick one more data for one-step update
    Xnew = Xtest[:1]; Ynew = Ytest[:1]
    #Xnew = Xtest[1:2]; Ynew = Ytest[1:2]
    Xtrain_plus = np.concatenate( (Xtrain, Xnew) )
    Ytrain_plus = np.concatenate( (Ytrain, Ynew) )    
    eff_Ytrain_plus = calc_eff_label(Xtrain_plus, Ytrain_plus, params)
    
    # total data (for estimating the true curve)
    Xtot = np.concatenate( (Xtrain, Xtest) )
    Ytot = np.concatenate( (Ytrain, Ytest) )
    
    # function estimated from offline, online, online (+1 sample), and offline (all data)
    Yoff = foff(Xtrain, Ytrain, Xs, params)
    Yon = fon(Xtrain, Ytrain, Xs, params)
    Yon_plus = fon(Xtrain_plus, Ytrain_plus, Xs, params)
    Yall = foff(Xtot, Ytot, Xs, params)

    Nplot = 10 # number of data points plotted in the figure
    idxs = rng.choice( range(len(Xtrain)), Nplot, replace=False )
    
    Np_ones = np.ones(( Nplot ))
    if data_type == 'toyGP':
        dxmin = 0.05 #0.04#0.03
    elif data_type == 'mcycle':
        dxmin = 3.0 
    
    # make sure that the plotted points are not super close to each other (for visualization purpose)
    while np.min( np.abs( np.outer(Xtrain[idxs], Np_ones) - np.outer( Np_ones, Xtrain[idxs]) + np.diag( 1e6*Np_ones ) ) ) < dxmin:
        idxs = rng.choice( range(len(Xtrain)), Nplot, replace=False )
    
    # plot effective target shifts
    fig1 = plt.figure(figsize=(6.4, 4.8))
    
    plt.plot(Xs, Yall, color='gray', ls='--')
    plt.plot(Xs, Yoff, color=ltype_clrs['off'])
    plt.plot(Xs, Yon, color=ltype_clrs['on'])
    
    for idx in idxs:
        plt.plot([Xtrain[idx], Xtrain[idx]], [Ytrain[idx], eff_Ytrain[idx]], color='k', lw=1.0 )
    
    plt.plot(Xtrain[idxs,0], Ytrain[idxs], '.', color=ltype_clrs['off'], ms=10.0)
    plt.plot(Xtrain[idxs,0], eff_Ytrain[idxs], 'x', color=ltype_clrs['on'], ms=10.0)    

    plt.show()
    
    fig1.savefig('figs/fig_' + data_type + '_target_shift_Ntrain' + str(params['Ntrain']) + '_sig2_' + f"{sig2:.3f}" + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + 'seed' + str(params['m_seed']) + '.pdf')
    
    
    # plot one sample update
    fig2 = plt.figure(figsize=(6.4, 4.8))
    
    plt.plot(Xs, Yall, color='gray', ls='--')
    plt.plot(Xs, Yon_plus, color=ltype_clrs['plus'])
    plt.plot(Xs, Yon, color=ltype_clrs['on'])
    
    for idx in idxs:
        plt.plot([Xtrain[idx], Xtrain[idx]], [eff_Ytrain_plus[idx], eff_Ytrain[idx]], color='k', lw=1.0 )
    
    #print( np.shape(idxs) )
    idxs_plus = np.append(idxs, len(Xtrain)) 
    plt.plot(Xtrain_plus[idxs_plus,0], eff_Ytrain_plus[idxs_plus], 'x', color=ltype_clrs['plus'], ms=10.0)
    plt.plot(Xtrain[idxs,0], eff_Ytrain[idxs], 'x', color=ltype_clrs['on'], ms=10.0)    

    plt.show()
    
    fig2.savefig('figs/fig_' + data_type + '_onestep_shift_Ntrain' + str(params['Ntrain']) + '_sig2_' + f"{sig2:.3f}" + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + 'seed' + str(params['m_seed'])  + '.pdf')
    
    # plot correction
    fig3 = plt.figure(figsize=(6.4, 4.8))
    
    plt.plot(Xs, Yall, color='gray', ls='--')
    plt.plot(Xs, Yoff, color=ltype_clrs['off'])
    plt.plot(Xs, Yon, color=ltype_clrs['on'])
    
    for idx in idxs:
        #plt.plot([Xtrain[idx], Xtrain[idx], Xtrain[idx]], [Ytrain[idx], eff_Ytrain[idx], Yc_train[idx]], color='k', lw=1.0 )
        plt.plot([Xtrain[idx], Xtrain[idx]], [Ytrain[idx], Yc_train[idx]], color='k', lw=1.0 )
        plt.plot([Xtrain[idx], Xtrain[idx]], [Ytrain[idx], eff_Ytrain[idx]], color='k', lw=1.0, ls='--' )
    
    plt.plot(Xtrain[idxs,0], Ytrain[idxs], '.', color=ltype_clrs['off'], ms=10.0, lw=2.0)
    plt.plot(Xtrain[idxs,0], eff_Ytrain[idxs], 'x', color=ltype_clrs['on'], ms=10.0, lw=2.0)    
    plt.plot(Xtrain[idxs,0], Yc_train[idxs], 'o', color=ltype_clrs['c'], ms=7.5, mfc='None', mec=ltype_clrs['c'], lw=2.0)   

    plt.show()
    
    fig3.savefig('figs/fig_' + data_type + '_target_shift_correction_Ntrain' + str(params['Ntrain']) + '_sig2_' + f"{sig2:.3f}" + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + 'seed' + str(params['m_seed'])  + '.pdf')



def plot_learning_curves(Nparts, off_errs, on_errs, on_simul_errs, mb_errs, mb_simul_errs, gms, etas, mb_sizes, params):
    V2L_cmap = mpl.colormaps['viridis'] 
    clrs = [V2L_cmap(0.0), V2L_cmap(0.4), V2L_cmap(0.8), V2L_cmap(0.9), V2L_cmap(1.0)]
    params_str = '_Ntrain' + str(params['Ntrain']) + '_sig2_' + f"{params['sig2']:.3f}"  + '_nlvl' + f"{params['noise_level']:.3f}" 
    
    
    fig1 = plt.figure( figsize=(4.8, 5.4) )
    for gmidx, gm in enumerate(gms):
        plt.plot(Nparts, off_errs[gm]['train'], color=clrs[gmidx], ls='--', lw=2.0)
        plt.plot(Nparts, off_errs[gm]['test'], color=clrs[gmidx], ls='-', lw=2.0, label=fr"$\gamma$ = {gm:.2f}")
    
    plt.legend()
    plt.xlim(0.0, 20.5)
    plt.show()
    fig1.savefig( 'figs/fig_off_curve_' + params_str + ".pdf" )
    
    
    fig2 = plt.figure( figsize=(4.8, 5.4) )
    for etidx, eta in enumerate(etas):
        plt.plot(Nparts, on_errs[etidx]['train'], color=clrs[etidx], ls='--')
        plt.plot(Nparts, on_errs[etidx]['test'], color=clrs[etidx], ls='-', label=fr"$\eta$ = {eta:.2f}")
        
        plt.plot(Nparts, on_simul_errs[etidx]['train'], 'o', color=clrs[etidx])
        plt.plot(Nparts, on_simul_errs[etidx]['test'], 's', color=clrs[etidx])

    plt.xlim(0.0, 20.5)
    plt.legend()
    plt.show()
    fig2.savefig( 'figs/fig_on_curve_' + params_str + ".pdf" )
    
    
    fig3 = plt.figure( figsize=(4.8, 5.4) )
    for mbidx, mb_size in enumerate(mb_sizes):
        mbNs = mb_errs[mbidx]['mbNs']

        plt.plot(mbNs, mb_errs[mbidx]['train'], color=clrs[mbidx], ls='--')
        plt.plot(mbNs, mb_errs[mbidx]['test'], color=clrs[mbidx], ls='-', label=f"mb = {mb_size}")
        
        plt.plot(mbNs, mb_simul_errs[mbidx]['train'], 'o', color=clrs[mbidx])
        plt.plot(mbNs, mb_simul_errs[mbidx]['test'], 's', color=clrs[mbidx])

    plt.xlim(0.0, 40.5)
    plt.legend()
    plt.show()
    fig3.savefig( 'figs/fig_minibatch_curve_' + params_str + ".pdf" )


