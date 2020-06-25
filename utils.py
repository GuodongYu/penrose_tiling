import json
import numpy as np
import sys
import os
from monty.json import jsanitize
from monty.io import zopen
import glob
import scipy.sparse as spsp
import scipy.linalg.lapack as spla
import numpy.linalg as npla

colors = ['blue', 'red', 'green', 'gray', 'cyan', 'magenta', 'yellow','black'] * 100

def e_Dirac(es, dos, eD_guess=0.0):
    es_rel = np.abs(np.array(es) - eD_guess)
    ind = np.argmin(es_rel)
    if dos[ind] > dos[ind+1]:
        di = 1
    elif dos[ind] > dos[ind-1]:
        di = -1
    else:
        return es[ind]

    while dos[ind+di] < dos[ind]:
        ind = ind + di
    return es[ind]

    
### density of states related functions ######
def efermi(eigs, nelec):
    n = int(nelec/2)
    if not nelec%2:
        return (eigs[n-1] + eigs[n])/2.0
    else:
        return eigs[n]

def get_dos(eig_file='eigen_values_vecs.json', save_to='dos_from_diag.json',
            nelec='norb', smear='G', sigma=0.01, de=0.001):
    """
    To get the density of states by diagnolization
    Args:
        eig_file: the json file saving the eigenvalues and eigenvectors
        smear:  'G': the gaussian method
                'L': the Lorentzian method
        sigma: the width to smearing the DOS
        de: the energy step for energy sampling
        plot: whether to plot the DOS
        save_to: only json or yaml file
    """
    data = json.load(zopen(eig_file))
    eigs = data['eigen_energies']
    if nelec == 'norb':
        nelec = len(eigs)
    ef = efermi(eigs, nelec)
    e0 = min(eigs)-0.1
    e1 = max(eigs)+0.1
    es = np.linspace(e0, e1, int((e1-e0)/de))
    dos = []
    def eigs_cut(e):
        e_min = e - 100*sigma
        e_max = e + 100*sigma
        es_ = np.array(eigs)
        es_ = es_[np.where(es_>=e_min)]
        es_ = es_[np.where(es_<=e_max)]
        return es_
    if smear == 'L':
        for e in es:
            summe = 0.
            es_ = eigs_cut(e)
            for ei in es_:
                tmp = np.pi*sigma/((e-ei)**2+sigma**2)
                summe = summe + tmp
            dos.append(summe)
    elif smear == 'G':
        for e in es:
            summe = 0.
            es_ = eigs_cut(e)
            for ei in es_:
                tmp = 1./( np.sqrt(2.*np.pi) * sigma ) * np.exp( -(e-ei)**2 / (2.*sigma**2) )
                summe = summe + tmp
            dos.append(summe)
    dos = {'energies':es, 'DOS':dos, 'efermi': ef}

    if save_to.split('.')[-1] == 'json':
        with open(save_to, 'w') as f:
            f.write(json.dumps(jsanitize(dos)))
    elif save_to.split('.')[-1] == 'yaml':
        with open(save_to, 'w') as f:
            yaml.dump(dos, f)

def get_multi_dos(eig_files):
    for f in eig_files:
        title = os.path.split(f)[-1]
        title = 'DOS_'+title
        get_dos(eig_file=f, nelec='norb', save_to=title)        

def plot_dos(dos_file, save_to='dos.pdf', xlim=None, ylim=None):
    """
    dos_file: the json file saving dos, which should be output by function get_dos
    """
    try:
        from matplotlib import pyplot as plt
    except:
        import matplotlib
        matplotlib.use('agg')
        from matplotlib import pyplot as plt
    dos = json.load(zopen(dos_file))
    plt.plot(dos['energies'], dos['DOS'])
    try:
        plt.axvline(dos['efermi'], linewidth=0.1, linestyle='dashed', color='black')
    except:
        pass
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    plt.xlabel('Energies (eV)')
    plt.ylabel('Density of states')
    plt.savefig(save_to)
    plt.clf()

def plot_multi_dos(dos_files, titles, save_to='multi_dos.pdf', xlim=None, ylim=None):
    """
    dos_files: the files list of list, such as [[f0, f1],[f2,f3]]
    the len(dos_files) subplots will be generated, and dos_files[i] will
    plot in the same subplot
    titles: the description title for all subplots
    len(dos_files) should be the same as len(titles)
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(dos_files), 1, sharex=True)
    for i in range(len(dos_files)):
        fs = dos_files[i]
        j = 0
        for f in fs:
            c = colors[j]
            dos = json.load(zopen(f))
            axs[i].plot(dos['energies'], dos['DOS'], color=c)
            axs[i].axvline(dos['efermi'], linewidth=0.1, linestyle='dashed', color='black')
            j = j + 1
        if ylim:
            axs[i].set_ylim(ylim)
        if xlim:
            axs[i].set_xlim(xlim)
        axs[i].text(0.1, 0.8, titles[i], transform=axs[i].transAxes)
    fig.text(0.5, 0.01, 'Energies(eV)', ha='center')
    fig.text(0.0, 0.5, 'Density of states', va='center',rotation='vertical')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.clf()
        

def get_avg_dos(dos_files, save_to='avg_dos.json'):
    n = len(dos_files)
    DOS = 0.
    for f in dos_files:
        dosi = json.load(zopen(f))
        DOS = DOS + np.array(dosi['DOS'])
    DOS = DOS/n
    dos = {}
    dos['energies'] = dosi['energies']
    dos['DOS'] = DOS
    #dos['efermi'] = 
   

############ get the locailzation of states #############################
def localization_calc(eig_file, eigval, width=0.2):
    """
    get the localization of eigstates around eigval
    args:
        eig_file:  the json file saving the eigvals and eigvecs
        eigval: calulate the localization of states aound eigval
                if eigval == 'Ef', it means one orbital induce one electron
        width: the energy window aournd eigval, within which the eigenstates are checked for localization
    """
    data = json.load(open(eig_file))
    eigvals = data['eigen_energies']
    eigvecs = data['eigen_states']
    if eigval == 'Ef':
        eigval = efermi(eigvals, len(eigvals))
    e0 = eigval - width/2.
    e1 = eigval + width/2.
    num = 0
    sum_4pow = 0.0
    dim = len(eigvals)
    for i,e in enumerate(eigvals):
        if e <= e1 and e >= e0:
            num = num +1
            vec_ = [eigvecs[j][i] for j in range(dim)]
            vec = np.array([complex(j) for j in vec_])
            charg = np.conj(vec)*vec
            tmp = np.linalg.norm(charg)
            sum_4pow = sum_4pow + tmp
    return sum_4pow/num


def localization_save(eig_files, width=0.2, from_file='', save_to='localization_out.json'):
    """
    calc and save the localization all systems with nelec = nval
    for vac, hopping disorder system
    Args:
        eig_files:  all eig_val_vec files
        width: the energy width
        from_file: the file save the already calculated localization
                           the file name without extension name is the key 
    """
    if from_file:
        localization = json.load(open(from_file))
    else:
        localization = {}
    for eig_file in eig_files:
        title = os.path.split(eig_file)[-1].split('.json')[0]
        if title in localization:
            continue
        try:
            loc = localization_calc(eig_file, 'Ef', width)
        except:
            continue
        localization[title] = loc
    with open(save_to, 'w') as f:
        f.write(json.dumps(jsanitize(localization)))
#################################################################################


################## IPR related functions #######################################
def get_IPR(eig_file, nelec='norb', save_to='IPR.json'):
    j = json.load(open(eig_file))
    energies = j['eigen_energies']
    ef = efermi(energies, len(energies))
    IPRS = []
    for i in range(len(energies)):
        vec = [complex(j['eigen_states'][k][i]) for k in range(len(energies))]
        IPR = np.linalg.norm(np.conjugate(vec)*np.array(vec))**2
        IPRS.append(IPR)
    out = {'efermi': ef, 'energies': energies, 'IPR': IPRS}
    with open(save_to, 'w') as f:
        f.write(json.dumps(jsanitize(out)))
    #return {'efermi': efermi, 'energies': energies, 'IPR': IPRS}

def get_multi_IPR(eig_files):
    for f in eig_files:
        title = os.path.split(f)[-1]
        get_IPR(f, save_to='IPR_' + title)

def plot_IPR(ipr_file, save_to='IPR.pdf', xlim=None, ylim=None):
    from matplotlib import pyplot as plt
    ipr = json.load(open(ipr_file))
    plt.scatter(ipr['energies'],ipr['IPR'], s=0.1)
    plt.axvline(ipr['efermi'], linewidth=0.1, linestyle='dashed',c='black')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel('Energies (eV)')
    plt.ylabel('IPR')
    plt.savefig(save_to)
    plt.clf()

def plot_multi_IPR(ipr_files, titles, save_to='multi_IPR.pdf', xlim=None, ylim=None):
    """
    ipr_files: list of list
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(ipr_files), sharex=True)
    for i in range(len(ipr_files)):
        j = 0
        for f in ipr_files[i]:
            c = colors[j]
            ipr = json.load(open(f))
            axs[i].scatter(ipr['energies'],ipr['IPR'], s=0.1, color=c, alpha=0.4)    
            axs[i].axvline(ipr['efermi'], linewidth=0.1, linestyle='dashed',c=c)
            j = j + 1
        axs[i].text(0.1, 0.8, titles[i], transform=axs[i].transAxes)
        if xlim:
            axs[i].set_xlim(xlim)
        if ylim:
            axs[i].set_ylim(ylim)
    #axs[len(ipr_files)-1].set_xlabel('Energies (eV)')
    fig.text(0.5, 0.01, 'Energies(eV)', ha='center')
    fig.text(0.0, 0.5, 'IPR', va='center',rotation='vertical')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.clf()    
######################################################################

def plot_eigstates(sample_file, eig_file, e_range, ipr_min, draw_size=5, fig_dpi=600, site_size=300):
    samp = json.load(zopen(sample_file))
    hops = []
    linews = []
    H = spsp.csr_matrix((samp['hop'], samp['indices'], samp['indptr']))
    H = H.tocoo()
    for i, j, hop in zip(H.row, H.col, H.data):
        if i > j:
            hops.append([[samp['site_x'][i], samp['site_y'][i]],
                         [samp['site_x'][j], samp['site_y'][j]]])
            linews.append(draw_size * npla.norm(hop))

    def plot_state(charg, fname, title):
        from matplotlib import pyplot as plt
        from matplotlib.cm import get_cmap
        import matplotlib.collections as mc

        fig, ax = plt.subplots()
        lines = mc.LineCollection(hops, linewidths=linews, colors='grey', zorder=1)
        ax.add_collection(lines)
        ax.set_aspect('equal')
        cm = get_cmap('Greys')
        ## get absolute square of wave function and sort

        ## make plot
        sc = ax.scatter(samp['site_x'], samp['site_y'], c=charg, s=charg*site_size, edgecolor='',cmap=cm)
        plt.axis('equal')
        plt.axis('off')
        plt.title(title)
        #if colorbar:
        #    plt.colorbar(sc)
        plt.draw()
        plt.savefig(fname, dpi=fig_dpi)
        plt.close()


    data = json.load(zopen(eig_file))
    eigenvals = np.array(data['eigen_energies'])
    eigenvecs = data['eigen_states']
    beg = np.abs(eigenvals-e_range[0])
    ind0 = np.argmin(beg)
    end = np.abs(eigenvals-e_range[1])
    ind1 = np.argmin(end)

    ns = range(ind0, ind1+1)
    for i in ns:
        ene = eigenvals[i]
        vec = np.array([complex(eigenvecs[k][i]) for k in range(len(eigenvals))])
        IPR = np.linalg.norm(np.conjugate(vec)*np.array(vec))**2
        if IPR < ipr_min:
            continue
        title = '$\\varepsilon=%seV IPR:%s$' % (ene, IPR)
        fname='charge_%i.pdf' % i
        charg = np.conj(vec)*vec
        plot_state(charg, fname, title)
    
def plot_AC(AC_file, save_to='ac.pdf', xlim=None, ylim=None):
    """
    dos_file: the json file saving dos, which should be output by function get_dos
    """
    try:
        from matplotlib import pyplot as plt
    except:
        import matplotlib
        matplotlib.use('agg')
        from matplotlib import pyplot as plt
    ac = json.load(zopen(AC_file))
    plt.plot(ac['omegas'], ac['AC'][0])
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    plt.xlabel('Omega ($\hbar\omega/t$)')
    plt.ylabel('AC conductivity ($\sigma/\sigma_0$)')
    plt.savefig(save_to)
    plt.clf()

def plot_multi_AC(ac_files, titles, save_to='multi_ac.pdf', xlim=None, ylim=None):
    """
    dos_files: the files list of list, such as [[f0, f1],[f2,f3]]
    the len(dos_files) subplots will be generated, and dos_files[i] will
    plot in the same subplot
    titles: the description title for all subplots
    len(dos_files) should be the same as len(titles)
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(ac_files), 1, sharex=True)
    for i in range(len(ac_files)):
        fs = ac_files[i]
        j = 0
        for f in fs:
            c = colors[j]
            ac = json.load(zopen(f))
            axs[i].plot(ac['omegas'], ac['AC'][0], color=c)
            j = j + 1
        if ylim:
            axs[i].set_ylim(ylim)
        if xlim:
            axs[i].set_xlim(xlim)
        axs[i].text(0.1, 0.8, titles[i], transform=axs[i].transAxes)
    fig.text(0.5, 0.01, '$\hbar\omega/t$', ha='center')
    fig.text(0.0, 0.5, 'AC conductivity ($\sigma/\sigma_0$)', va='center',rotation='vertical')
    plt.tight_layout()
    plt.savefig(save_to)
    plt.clf()

def plot_AC_comparison(ac_files, legends, save_to='AC_comparison.pdf', xlim=None, ylim=None):
    from matplotlib import pyplot as plt
    for i in range(len(ac_files)):
        f = ac_files[i]
        ac = json.load(open(f))
        plt.plot(ac['omegas'], ac['AC'][0], label=legends[i])
    plt.legend()
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xlabel('$\hbar\omega/t$')
    plt.ylabel('AC conductivity ($\sigma/\sigma_0$)')
    plt.savefig(save_to)
    plt.clf()

def plot_DOS(dos_file, save_to='dos.pdf', xlim=None, ylim=None):
    from matplotlib import pyplot as plt
    dos = json.load(open(dos_file))
    plt.plot(dos['energies'], dos['DOS'])
    plt.axvline(dos['efermi'], linestyle='dashed', linewidth=0.1)
    plt.xlabel('Energies (eV)')
    plt.ylabel('Density of states')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.savefig(save_to)
    plt.clf()

def plot_DOS_comporison(dos_files, legends, save_to='dos_comparison.pdf', xlim=None, ylim=None):
    from matplotlib import pyplot as plt
    for i in range(len(dos_files)):
        f = dos_files[i]
        if f.split('.')[-1] == 'json':
            dos = json.load(open(f))
            ef = dos['efermi']
            es = dos['energies']
            DOS = dos['DOS']
            #eD = e_Dirac(es, DOS, ef)
            eD = 0.905
        elif f.split('.')[-1] in ['txt','dat']:
            dos = np.loadtxt(f)
            es = dos[:,0]
            DOS = dos[:,1]
            #eD = e_Dirac(es, DOS, 0.)
            eD = 0.905 - 0.7968
        if legends[i]=='Graphene':
            eD = eD + 0.7968 - 0.905 
        es = np.array(es)-eD
        plt.plot(es, DOS, label=legends[i], linewidth=1.0)
        #plt.plot(es, DOS, label=legends[i])
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Energies (eV)')
    plt.ylabel('Density of states')
    plt.savefig(save_to)
    plt.clf()
    
def plot_Hall_cond(Hall_cond_file, save_to='Hall_cond.pdf', xlim=None, ylim=None):
    from matplotlib import pyplot as plt
    hc = json.load(open(Hall_cond_file))
    plt.plot(hc['energies'], hc['cond'])
    plt.xlabel('Energies (eV)')
    plt.ylabel('Hall conductivity ($e^2/h$)')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.grid('True')
    plt.savefig(save_to)
    plt.clf()

def plot_Hall_cond_comporison(Hall_cond_files, legends, save_to='Hall_cond_comparison.pdf', xlim=None, ylim=None):
    from matplotlib import pyplot as plt
    for i in range(len(Hall_cond_files)):
        f = Hall_cond_files[i]
        hc = json.load(open(f))
        plt.plot(hc['energies'], hc['cond'], label=legends[i])
        txt={}
        num = 0 
        for j in range(len(hc['cond'])):
            hcj = hc['cond'][j]
            ej = hc['energies'][j] 
            if num == 0:
                value = hcj
                txt[value]=[ej]
                num = 1
            elif abs(hcj-value) <= 0.5:
                txt[value].append(ej)
                num = num + 1
            else:
                value = hcj
                txt[value]=[ej]
                num = 1
        for k in txt:
            if len(txt[k]) > 6 and np.var(txt[k])<0.04: 
                val = round(k, 2)
                ene = np.mean(txt[k])
                plt.text(ene, val, str(val))
                plt.scatter(ene, val)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.xlabel('Energies (eV)')
    plt.ylabel('Hall_condcutivity ($e^2/h$)')
    plt.savefig(save_to)
    plt.clf()
