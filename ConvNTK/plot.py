#
# Effective label shift and correction: MNIST-NTK Example
#
# Plotting
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("ggplot")
plt.rcParams.update({'font.size':16})

cmap = mpl.colormaps['tab10']
ltype_clrs = {'off': cmap(0.05), 'on': cmap(0.15), 'c': cmap(0.25), 'c_iter': cmap(0.35)}   
    
def plot_errors_perfs(xs, errors, perfs, filename):
    
    fig = plt.figure(figsize=(8.4, 4.8))
    
    plt.subplot(1,2,1)
    plt.plot(xs, errors, 'o-')
    plt.loglog()
    
    plt.subplot(1,2,2)
    plt.plot(xs, perfs, 'o-')
    plt.semilogx()
    
    plt.show()
    
    fig.savefig( filename )


def plot_conv_ntk_gm_dep(hy_params):
    ikmax = hy_params['ikmax']
    Ntrain, Ntest = hy_params['Ntrain'], hy_params['Ntest']
    
    for ik in range(ikmax):
        params_str = 'ttype_' + str(hy_params['task_type']) + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_ik' + str(ik)
        fgm_str = 'data/conv_ntk_gm_dep_' + params_str + '.csv'
        
        df = pd.read_csv(fgm_str, names = ['gm', 'error', 'perf'])
        
        if ik == 0:
            gms = df['gm'].to_numpy()
            errors = np.zeros(( len(gms), ikmax ))
            perfs = np.zeros(( len(gms), ikmax ))
        errors[:, ik] = df['error'].to_numpy()
        perfs[:, ik] = df['perf'].to_numpy()
    
    mean_errors = np.mean(errors, axis=1)
    ste_errors = np.std(errors, axis=1)/np.sqrt(ikmax)
    
    mean_perfs = np.mean(perfs, axis=1)
    ste_perfs = np.std(perfs, axis=1)/np.sqrt(ikmax)
    
    fig = plt.figure(figsize=(9.6, 4.8))
    
    plt.subplot(1,2,1)
    plt.errorbar(gms, mean_errors, ste_errors, elinewidth=2.0, capsize=2.0, color=ltype_clrs['off'])
    plt.loglog()
    #plt.ylim(0.009, 0.11)
    
    plt.subplot(1,2,2)
    plt.errorbar(gms, mean_perfs, ste_perfs, elinewidth=2.0, capsize=2.0, color=ltype_clrs['off'])
    plt.semilogx()
    plt.ylim(0.0, 1.0)
    
    plt.show()
    fig.savefig( 'figs/fig_conv_ntk_gm_dep_ttype_' + str(hy_params['task_type']) + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest)\
                    + '_ikm' + str(ikmax) + '.pdf' )


def plot_conv_ntk_eta_dep(hy_params):
    ikmax = hy_params['ikmax']
    Ntrain, Ntest = hy_params['Ntrain'], hy_params['Ntest']
    
    for ik in range(ikmax):
        params_str = 'ttype_' + str(hy_params['task_type']) + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_ik' + str(ik)
        feta_str = 'data/conv_ntk_eta_dep_' + params_str + '.csv'
        
        df = pd.read_csv(feta_str, names = ['eta', 'error', 'perf'])
        
        if ik == 0:
            etas = df['eta'].to_numpy()
            errors = np.zeros(( len(etas), ikmax ))
            perfs = np.zeros(( len(etas), ikmax ))
        errors[:, ik] = df['error'].to_numpy()
        perfs[:, ik] = df['perf'].to_numpy()
    
    mean_errors = np.mean(errors, axis=1)
    ste_errors = np.std(errors, axis=1)/np.sqrt(ikmax)
    
    mean_perfs = np.mean(perfs, axis=1)
    ste_perfs = np.std(perfs, axis=1)/np.sqrt(ikmax)
    
    fig = plt.figure(figsize=(9.6, 4.8))
    
    plt.subplot(1,2,1)
    plt.errorbar(etas, mean_errors, ste_errors, elinewidth=2.0, capsize=2.0, color=ltype_clrs['on'])
    plt.loglog()
    plt.ylim(0.01, 10)
    
    plt.subplot(1,2,2)
    plt.errorbar(etas, mean_perfs, ste_perfs, elinewidth=2.0, capsize=2.0, color=ltype_clrs['on'])
    plt.semilogx()
    plt.ylim(0.0, 1.0)
    
    plt.show()
    fig.savefig( 'figs/fig_conv_ntk_eta_dep_ttype_' + str(hy_params['task_type']) + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest)\
                    + '_ikm' + str(ikmax) + '.pdf' )
    

# plotting the label shift in the direction of y3 and y5
def plot_label_shift_35( rng, Ytrue, Yeff, Yc, hy_params, with_correction, Nplot=10 ):
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']

    idx3 = np.where( np.argmax(Ytrue, axis=0) == 3 )[0]
    idx5 = np.where( np.argmax(Ytrue, axis=0) == 5 )[0]
    
    Nplot = 10
    if len(idx3) > Nplot:
        iidxs = rng.choice( range( len(idx3) ), Nplot, replace=False )
        idx3 = idx3[iidxs]
    if len(idx5) > Nplot:
        iidxs = rng.choice( range( len(idx5) ), Nplot, replace=False )
        idx5 = idx5[iidxs]
    
    fig = plt.figure(figsize=(6.0, 4.8))
    
    for idx in idx3:
        plt.plot( [Ytrue[3,idx3], Yeff[3,idx3]], [Ytrue[5,idx3], Yeff[5,idx3]], color='C0', lw=0.5 )
    plt.scatter( Ytrue[3,idx3], Ytrue[5,idx3], marker='o', color='C0' )
    plt.scatter( Yeff[3,idx3], Yeff[5,idx3], marker='x', color='C0' )
    
    for idx in idx5:
        plt.plot( [Ytrue[3,idx5], Yeff[3,idx5]], [Ytrue[5,idx5], Yeff[5,idx5]], color='C1', lw=0.5 )
    plt.scatter( Ytrue[3,idx5], Ytrue[5,idx5], marker='o', color='C1' )
    plt.scatter( Yeff[3,idx5], Yeff[5,idx5], marker='x', color='C1' )
    
    if with_correction:
        for idx in idx3:
            plt.plot( [Ytrue[3,idx3], Yc[3,idx3]], [Ytrue[5,idx3], Yc[5,idx3]], color='C0', lw=0.5 )
        plt.scatter( Yc[3,idx3], Yc[5,idx3], marker='o', fc='white', ec='C0', lw=1.0 )
        
        for idx in idx5:
            plt.plot( [Ytrue[3,idx5], Yc[3,idx5]], [Ytrue[5,idx5], Yc[5,idx5]], color='C1', lw=0.5 )    
        plt.scatter( Yc[3,idx5], Yc[5,idx5], marker='o', fc='white', ec='C1', lw=1.0 )  
        
    plt.show()
    
    gm, eta = hy_params['gm'], hy_params['eta']
    if with_correction:
        fig.savefig( 'figs/plot_label_shift_correction_35_Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + '_Nplot' + str(Nplot) + '.pdf' )
    else:
        fig.savefig( 'figs/plot_label_shift_35_Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + '_Nplot' + str(Nplot) + '.pdf' )


# plotting the label shift in the direction of y0, ..., y4
def plot_label_shift_0to4( rng, Ytrue, Yeff, Yc, hy_params, with_correction, Nplot=10 ):
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']
    
    cmap = mpl.colormaps['Dark2']
    normalized_values = np.arange(0.05, 0.96, 0.1)
    digit_clrs = cmap(normalized_values)
    
    #numbers = [0, 1, 2, 3, 4]
    numbers = [0, 1, 2]
    s_idxs = []
    for number in numbers:
        num_idx = np.where( np.argmax(Ytrue, axis=0) == number )[0]
        if len(num_idx) > Nplot:
            iidxs = rng.choice( range( len(num_idx) ), Nplot, replace=False )
            s_idxs.append( num_idx[iidxs] )
        else:
            s_idxs.append( num_idx )
    
    
    #fig = plt.figure(figsize=(10.8, 9.6))
    fig = plt.figure(figsize=(7.2, 6.4))
    for i in numbers:
        for j in numbers:
            if j > i:
                plt.subplot( 2, 2, i*2 + j )
                for idx in s_idxs[i]:
                    plt.plot( [Ytrue[i,idx], Yeff[i,idx]], [Ytrue[j,idx], Yeff[j,idx]], color=digit_clrs[i], lw=0.5 )
                iidx = s_idxs[i]
                plt.scatter( Ytrue[i,iidx], Ytrue[j,iidx], marker='o', color=digit_clrs[i] )
                plt.scatter( Yeff[i,iidx], Yeff[j,iidx], marker='x', color=digit_clrs[i] )
                
                for idx in s_idxs[j]:
                    plt.plot( [Ytrue[i,idx], Yeff[i,idx]], [Ytrue[j,idx], Yeff[j,idx]], color=digit_clrs[j], lw=0.5 )
                jidx = s_idxs[j]
                plt.scatter( Ytrue[i,jidx], Ytrue[j,jidx], marker='o', color=digit_clrs[j] )
                plt.scatter( Yeff[i,jidx], Yeff[j,jidx], marker='x', color=digit_clrs[j] )
    plt.tight_layout()
    plt.show()
    
    gm, eta = hy_params['gm'], hy_params['eta']
    fig.savefig( 'figs/plot_label_shift_0to2_Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + '_Nplot' + str(Nplot) + '.pdf' )

    if with_correction:
        #fig = plt.figure(figsize=(10.0, 9.6))
        fig = plt.figure(figsize=(7.2, 6.4))
        for i in numbers:
            for j in numbers:
                if j > i:
                    #plt.subplot(4, 4, i*4 + j)
                    plt.subplot(2, 2, i*2 + j)
                    for idx in s_idxs[i]:
                        plt.plot( [Ytrue[i,idx], Yc[i,idx]], [Ytrue[j,idx], Yc[j,idx]], color=digit_clrs[i], lw=0.5 )
                        plt.plot( [Ytrue[i,idx], Yeff[i,idx]], [Ytrue[j,idx], Yeff[j,idx]], color=digit_clrs[i], lw=0.5 )
                    
                    iidx = s_idxs[i]
                    plt.scatter( Yc[i,iidx], Yc[j,iidx], marker='o', color=digit_clrs[i], fc='None', ec=digit_clrs[i], lw=1.5 )
                    plt.scatter( Ytrue[i,iidx], Ytrue[j,iidx], marker='o', color=digit_clrs[i] )
                    plt.scatter( Yeff[i,iidx], Yeff[j,iidx], marker='x', color=digit_clrs[i] )
                    
                    for idx in s_idxs[j]:
                        plt.plot( [Ytrue[i,idx], Yc[i,idx]], [Ytrue[j,idx], Yc[j,idx]], color=digit_clrs[j], lw=0.5 )
                        plt.plot( [Ytrue[i,idx], Yeff[i,idx]], [Ytrue[j,idx], Yeff[j,idx]], color=digit_clrs[j], lw=0.5 )
                    jidx = s_idxs[j]
                    plt.scatter( Yc[i,jidx], Yc[j,jidx], marker='o', color=digit_clrs[j], fc='None', ec=digit_clrs[j], lw=1.5 )
                    plt.scatter( Ytrue[i,jidx], Ytrue[j,jidx], marker='o', color=digit_clrs[j] )
                    plt.scatter( Yeff[i,jidx], Yeff[j,jidx], marker='x', color=digit_clrs[j] )
        plt.tight_layout()
        plt.show()
        
        gm, eta = hy_params['gm'], hy_params['eta']
        fig.savefig( 'figs/plot_label_shift_correction_0to2_Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                        + '_eta' + f"{eta:.3f}" + '_Nplot' + str(Nplot) + '.pdf' )
        


def plot_learning_curve(hy_params):
    task_type = hy_params['task_type']
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']
    gm, eta, ikmax = hy_params['gm'], hy_params['eta'], hy_params['ikmax']
    
    ltypes = ['off', 'on', 'c', 'c_iter']
    dN = 16
    dNiter = hy_params['dNiter']
    Ns = range(dN, Ntrain+1, dN)
    Nlen = len(Ns)
    
    errors = {'off': np.zeros((Nlen, ikmax)), 'on': np.zeros((Nlen, ikmax)), 'c': np.zeros((Nlen, ikmax)), 'c_iter': np.zeros((Nlen, ikmax))}
    perfs = {'off': np.zeros((Nlen, ikmax)), 'on': np.zeros((Nlen, ikmax)), 'c': np.zeros((Nlen, ikmax)), 'c_iter': np.zeros((Nlen, ikmax))}

    for ik in range(ikmax):
        params_str = 'ttype_' + task_type + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + '_dNiter' + str(dNiter) + '_ik' + str(ik)
        flc_str = 'data/conv_ntk_learn_curve_' + params_str + '.csv'
        df = pd.read_csv(flc_str, names= ['N', 'off_error', 'off_perf', 'on_error', 'on_perf', 'c_error', 'c_perf', 'c_iter_error', 'c_iter_perf'])

        for ltype in ltypes:
            errors[ltype][:,ik] = df[ltype + '_error'].to_numpy()
            perfs[ltype][:,ik] = df[ltype + '_perf'].to_numpy()
    
    if task_type == 'class_inc':
        gm_cin = 1.0
        eta_cin = 0.3
        c_ltypes = ['c_iter']
        for ik in range(ikmax):
            params_str = 'ttype_' + task_type + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm_cin:.3f}"\
                        + '_eta' + f"{eta_cin:.3f}" + '_dNiter' + str(dNiter) + '_ik' + str(ik)
            flc_str = 'data/conv_ntk_learn_curve_' + params_str + '.csv'
            df = pd.read_csv(flc_str, names= ['N', 'off_error', 'off_perf', 'on_error', 'on_perf', 'c_error', 'c_perf', 'c_iter_error', 'c_iter_perf'])
            
            for ltype in c_ltypes:
                errors[ltype][:,ik] = df[ltype + '_error'].to_numpy()
                perfs[ltype][:,ik] = df[ltype + '_perf'].to_numpy()
    
    fig1 = plt.figure(figsize=(5.4, 4.8))
    
    for ltype in ltypes:
        mean_errors = np.mean(errors[ltype], axis=1)
        ste_errors = np.std(errors[ltype], axis=1)/np.sqrt(ikmax)
        
        plt.fill_between( Ns, mean_errors+ste_errors, mean_errors-ste_errors, color=ltype_clrs[ltype], alpha=0.2 )
        plt.plot( Ns, mean_errors, color=ltype_clrs[ltype], lw=2.0)
    
    plt.xlim(-15, 1040)
    plt.ylim(0.01, 100.0)
    plt.semilogy()
    plt.show()
    
    fig1.savefig('figs/fig_error_curve_offonc_Ntrain' + 'ttype_' + hy_params['task_type'] + str(Ntrain) + '_Ntest' + str(Ntest)\
                    + '_gm' + f"{gm:.3f}" + '_eta' + f"{eta:.3f}" + '_dNiter' + f"{dNiter}" + '_ikm' + str(ikmax) + '.pdf' )
  
    
    fig2 = plt.figure(figsize=(5.4, 4.8))
    for ltype in ltypes:
        mean_perfs = np.mean(perfs[ltype], axis=1)
        ste_perfs = np.std(perfs[ltype], axis=1)/np.sqrt(ikmax)
        
        plt.fill_between( Ns, mean_perfs+ste_perfs, mean_perfs-ste_perfs, color=ltype_clrs[ltype], alpha=0.2 )
        if ltype == 'off':
            plt.plot( Ns, mean_perfs, color=ltype_clrs[ltype], lw=2.0, ls='--')
        else:   
            plt.plot( Ns, mean_perfs, color=ltype_clrs[ltype], lw=2.0)

    plt.xlim(-15, 1040)
    plt.ylim(0.0, 1.0)

    plt.show()
    if task_type == 'class_inc':
        fig2.savefig('figs/fig_error_curve_offonc_Ntrain' + 'ttype_' + hy_params['task_type'] + str(Ntrain) + '_Ntest' + str(Ntest)\
                    + '_gm' + f"{gm:.3f}" + f"-{gm_cin:.3f}" + '_eta' + f"{eta:.3f}" + f"-{eta_cin:.3f}" + '_dNiter' + f"{dNiter}" + '_ikm' + str(ikmax) + '.pdf' )
    else:
        fig2.savefig('figs/fig_error_curve_offonc_Ntrain' + 'ttype_' + hy_params['task_type'] + str(Ntrain) + '_Ntest' + str(Ntest)\
                    + '_gm' + f"{gm:.3f}" + '_eta' + f"{eta:.3f}" + '_dNiter' + f"{dNiter}" + '_ikm' + str(ikmax) + '.pdf' )
    

def plot_dNiter_gm_dependence(hy_params):
    Ntrain, Ntest, Nb = hy_params['Ntrain'], hy_params['Ntest'], hy_params['Nb']
    gm, eta, ikmax = hy_params['gm'], hy_params['eta'], hy_params['ikmax']
    
    ltypes = ['off', 'on', 'c', 'c_iter']
    dN = 16
    Ns = range(dN, Ntrain+1, dN)
    Nlen = len(Ns)
    
    dNiters = [4, 16, 64, 256]
    
    dN_perfs = np.zeros(( len(dNiters), Nlen, ikmax ))
    for dNidx, dNiter in enumerate(dNiters):
        hy_params['dNiter'] = dNiter
    
        errors = {'off': np.zeros((Nlen, ikmax)), 'on': np.zeros((Nlen, ikmax)), 'c': np.zeros((Nlen, ikmax)), 'c_iter': np.zeros((Nlen, ikmax))}
        perfs = {'off': np.zeros((Nlen, ikmax)), 'on': np.zeros((Nlen, ikmax)), 'c': np.zeros((Nlen, ikmax)), 'c_iter': np.zeros((Nlen, ikmax))}
        
        for ik in range(ikmax):
            params_str = 'ttype_' + hy_params['task_type'] + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                        + '_eta' + f"{eta:.3f}" + '_dNiter' + str(dNiter) + '_ik' + str(ik)
            flc_str = 'data/conv_ntk_learn_curve_' + params_str + '.csv'
            
            df = pd.read_csv(flc_str, names= ['N', 'off_error', 'off_perf', 'on_error', 'on_perf', 'c_error', 'c_perf', 'c_iter_error', 'c_iter_perf'])
            for ltype in ltypes:
                errors[ltype][:,ik] = df[ltype + '_error'].to_numpy()
                perfs[ltype][:,ik] = df[ltype + '_perf'].to_numpy()
        
        dN_perfs[dNidx,:,:] = perfs['c_iter'][:,:]
    
    p_cmap = mpl.colormaps['plasma'] 
    clrs = [p_cmap(0.0), p_cmap(0.27), p_cmap(0.54), p_cmap(0.81)]
    
    fig1 = plt.figure(figsize=(5.4, 4.8))
    for dNidx, dNiter in enumerate(dNiters):
        hy_params['dNiter'] = dNiter
        mean_perfs = np.mean(dN_perfs[dNidx], axis=1)
        ste_perfs = np.std(dN_perfs[dNidx], axis=1)/np.sqrt(ikmax)
        
        plt.fill_between( Ns, mean_perfs+ste_perfs, mean_perfs-ste_perfs, color=clrs[dNidx], alpha=0.2 )
        plt.plot( Ns, mean_perfs, color=clrs[dNidx], lw=2.0, label=f'{dNiter}')

    plt.xlim(-15, 1040)
    plt.ylim(0.0, 1.0)

    plt.legend()
    plt.show()
    fig1.savefig('figs/fig_error_curve_dNiter_dep_Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + '_ikm' + str(ikmax) + '.pdf' )
                    
    
    gms = [0.01, 0.1, 1.0, 10.0]
    clrs = [p_cmap(0.0), p_cmap(0.27), p_cmap(0.54), p_cmap(0.81)]
    
    dNiter = 16
    hy_params['dNiter'] = dNiter
    
    gm_perfs = np.zeros(( len(gms), Nlen, ikmax ))
    for gmidx, gm in enumerate(gms):
        hy_params['gm'] = gm
    
        errors = {'off': np.zeros((Nlen, ikmax)), 'on': np.zeros((Nlen, ikmax)), 'c': np.zeros((Nlen, ikmax)), 'c_iter': np.zeros((Nlen, ikmax))}
        perfs = {'off': np.zeros((Nlen, ikmax)), 'on': np.zeros((Nlen, ikmax)), 'c': np.zeros((Nlen, ikmax)), 'c_iter': np.zeros((Nlen, ikmax))}
        
        for ik in range(ikmax):
            params_str = 'ttype_' + hy_params['task_type'] + 'Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_gm' + f"{gm:.3f}"\
                        + '_eta' + f"{eta:.3f}" + '_dNiter' + str(dNiter) + '_ik' + str(ik)
            flc_str = 'data/conv_ntk_learn_curve_' + params_str + '.csv'
            
            df = pd.read_csv(flc_str, names= ['N', 'off_error', 'off_perf', 'on_error', 'on_perf', 'c_error', 'c_perf', 'c_iter_error', 'c_iter_perf'])
            for ltype in ltypes:
                errors[ltype][:,ik] = df[ltype + '_error'].to_numpy()
                perfs[ltype][:,ik] = df[ltype + '_perf'].to_numpy()
        
        gm_perfs[gmidx,:,:] = perfs['c'][:,:]
    
    p_cmap = mpl.colormaps['plasma'] 
    clrs = [p_cmap(0.0), p_cmap(0.27), p_cmap(0.54), p_cmap(0.81)]
    
    fig2 = plt.figure(figsize=(5.4, 4.8))
    for gmidx, gm in enumerate(gms):
        hy_params['gm'] = gm
        mean_perfs = np.mean(gm_perfs[gmidx], axis=1)
        ste_perfs = np.std(gm_perfs[gmidx], axis=1)/np.sqrt(ikmax)
        
        plt.fill_between( Ns, mean_perfs+ste_perfs, mean_perfs-ste_perfs, color=clrs[gmidx], alpha=0.2 )
        plt.plot( Ns, mean_perfs, color=clrs[gmidx], lw=2.0, label=f'{gm:.3f}')

    plt.xlim(-15, 1040)
    plt.ylim(0.0, 1.0)

    plt.legend()
    plt.show()
    fig2.savefig('figs/fig_error_curve_gm_dep_Ntrain' + str(Ntrain) + '_Ntest' + str(Ntest) + '_dNiter' + f"{gm:.3f}"\
                    + '_eta' + f"{eta:.3f}" + '_ikm' + str(ikmax) + '.pdf' )



if __name__ == "__main__":
    
    #hyper parameters
    hy_params = {
        'Ntrain': 1024, # [256, 512, 1024]
        'Ntest': 256,
        'Nb': 16, # the block size for memory efficiency
        
        'gm': 0.01, 
        'eta': 1.0,
        'dNiter': 16,
        
        'task_type': 'vanilla',
        
        'ikmax': 10, # maximum number of seeds
    }
    
    # parameter dependence
    #for ttype in ['vanilla', 'class_inc']: #'vanilla', 'class_inc', 'task_inc']:
    #    hy_params['task_type'] = ttype
    #    plot_conv_ntk_gm_dep(hy_params)
    #    plot_conv_ntk_eta_dep(hy_params)
    
    # plot performance comparison across learning rules
    plot_learning_curve(hy_params)
    
    # plot dNiter and gm dependence of the learning curves
    plot_dNiter_gm_dependence(hy_params)
    
    # plotting the results for class-incremental setting
    hy_params['task_type'] = 'class_inc'
    hy_params['dNiter'] = 16
    hy_params['gm'] = 0.001
    hy_params['eta'] = 0.001
    plot_learning_curve(hy_params)
    
    

    
